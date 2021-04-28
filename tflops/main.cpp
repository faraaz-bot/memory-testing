/*
 * rocFFT teraflop benchmark
 */

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>
#include <vector>

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <rocfft.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_FFTW
#include <fftw3.h>
#endif

#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return;

#define HIP_CHECK0(r)   \
    if(r != hipSuccess) \
        return 0;

#define MPI_CHECK(r)     \
    if(r != MPI_SUCCESS) \
        return;

//
// Helpers
//

float average(std::vector<float> x)
{
    return accumulate(x.cbegin(), x.cend(), 0.0) / x.size();
}

template <typename T>
std::vector<T> random_vector(size_t n)
{
    std::vector<T>                         x(n);
    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
    for(size_t i = 0; i < n; ++i)
    {
        // always use double for dis(gen), save as real type
        x[i].x = dis(gen);
        x[i].y = dis(gen);
    }
    return x;
}

template <typename T>
double compare(std::vector<T> const& z1, std::vector<T> const& z2)
{
    double diff = 0.0;
    double norm = 0.0;
    for(size_t n = 0; n < z1.size(); ++n)
    {
        double dx = z1[n].x - z2[n].x;
        double dy = z1[n].y - z2[n].y;
        diff += dx * dx + dy * dy;
        norm += z1[n].x * z1[n].x + z1[n].y * z1[n].y;
    }
    return sqrt(diff) / sqrt(norm);
}

//
// Timers
//

struct GPUTimer
{
    hipEvent_t start, stop;

    GPUTimer()
    {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }

    ~GPUTimer()
    {
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void tic()
    {
        HIP_CHECK(hipEventRecord(start, 0));
    }

    void toc()
    {
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
    }

    float elapsed()
    {
        float elapsed;
        HIP_CHECK0(hipEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }
};

//
// GPU buffer
//

class GPUBuffer
{
public:
    GPUBuffer() {}

    GPUBuffer(size_t nbytes)
    {
        allocate(nbytes);
    }

    ~GPUBuffer()
    {
        if(nbytes > 0)
            hipFree(buf);
    }

    void* data()
    {
        return buf;
    }

    void allocate(size_t nbytes_)
    {
        nbytes   = nbytes_;
        auto res = hipMalloc(&buf, nbytes);
        if(res != hipSuccess)
        {
            throw std::runtime_error("hipMalloc failed");
        }
    }

    template <typename T>
    void copy_from_host(std::vector<T> const& src)
    {
        if(nbytes == 0)
            allocate(src.size() * sizeof(T));
        auto res = hipMemcpy(buf, src.data(), nbytes, hipMemcpyHostToDevice);
        if(res != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy to device failed");
        }
    }

    template <typename T>
    void copy_to_host(std::vector<T>& dst)
    {
        auto res = hipMemcpy(dst.data(), buf, nbytes, hipMemcpyDeviceToHost);
        if(res != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy to host failed");
        }
    }

private:
    size_t nbytes = 0;
    void*  buf    = nullptr;
};

//
// FFTW backed FFT
//

#ifdef USE_FFTW
std::vector<hipDoubleComplex> fft_fftw(std::vector<hipDoubleComplex> const& x, int nx, int nbatch)
{
    std::vector<hipDoubleComplex> z = x;
    // clang-format off
    auto p = fftw_plan_many_dft(1, &nx, nbatch,
                                (fftw_complex*) z.data(), nullptr, 1, nx,
                                (fftw_complex*) z.data(), nullptr, 1, nx,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    fftw_execute(p);
    fftw_destroy_plan(p);
    return z;
}
#endif

#ifdef USE_FFTW
std::vector<hipComplex> fft_fftw(std::vector<hipComplex> const& x, int nx, int nbatch)
{
    std::vector<hipComplex> z = x;
    // clang-format off
    auto p = fftwf_plan_many_dft(1, &nx, nbatch,
                                (fftwf_complex*) z.data(), nullptr, 1, nx,
                                (fftwf_complex*) z.data(), nullptr, 1, nx,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    return z;
}
#endif

//
// GPU FFT
//

template <typename T>
std::tuple<float, std::vector<T>> fft_gpu(std::vector<T> const& input, size_t nx, size_t nbatch)
{
    GPUBuffer      gpu_inout, gpu_work;
    std::vector<T> output(input.size());

    // Create FFT plan
    rocfft_plan           plan      = nullptr;
    rocfft_precision      precision = rocfft_precision_double;
    rocfft_execution_info info      = nullptr;

    if(typeid(T) == typeid(hipComplex))
        precision = rocfft_precision_single;

    size_t length[1] = {nx};
    rocfft_plan_create(&plan,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_forward,
                       precision,
                       1,
                       length,
                       nbatch,
                       nullptr);

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    rocfft_plan_get_work_buffer_size(plan, &work_buf_size);
    if(work_buf_size)
    {
        rocfft_execution_info_create(&info);
        gpu_work.allocate(work_buf_size);
        rocfft_execution_info_set_work_buffer(info, gpu_work.data(), work_buf_size);
    }

    // Warm-up FFT (copy out for verification)
    gpu_inout.copy_from_host(input);
    void* x_d = gpu_inout.data();
    rocfft_execute(plan, &x_d, nullptr, info);
    gpu_inout.copy_to_host(output);

    // Timed FFT
    gpu_inout.copy_from_host(input);

    GPUTimer timer;
    timer.tic();
    rocfft_execute(plan, &x_d, nullptr, info);
    timer.toc();

    // Clean up
    if(info)
        rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);

    return {timer.elapsed(), move(output)};
}

//
// Some tests!
//
template <typename T>
void test1d(int n, int nbatch)
{
#ifdef USE_MPI
    int mpi_rank = -1, mpi_size = -1, devices = -1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    HIP_CHECK(hipGetDeviceCount(&devices));

    int device = mpi_rank % devices;
    HIP_CHECK(hipSetDevice(device));

    bool echo = mpi_rank == 0;

    if(mpi_size > 0)
    {
        if(mpi_rank == 0)
            nbatch = nbatch - (mpi_size - 1) * (nbatch / mpi_size);
        else
            nbatch = nbatch / mpi_size;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for(int r = 0; r < mpi_size; ++r)
    {
        if(r == mpi_rank)
        {
            std::cout << "MPI rank/size:    " << mpi_rank << "/" << mpi_size << std::endl;
            std::cout << "GPU device:       " << device << "/" << devices << std::endl
                      << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
    bool echo = true;
#endif

    double GB = double(n * nbatch * sizeof(T)) / 1000 / 1000 / 1000;

    if(echo)
    {
        std::cout << "1D transform:     forward complex-to-complex" << std::endl;
        if(typeid(T) == typeid(hipComplex))
            std::cout << "1D precision:     single" << std::endl;
        else
            std::cout << "1D precision:     double" << std::endl;
        std::cout << "1D input length:  " << n << std::endl;
        std::cout << "1D input batch:   " << nbatch << std::endl;
        std::cout << "1D input size:    " << GB << " GB" << std::endl;
    }

    auto x        = random_vector<T>(n * nbatch);
    auto [t2, z2] = fft_gpu(x, n, nbatch);

#ifdef USE_FFTW
    if(echo)
    {
        auto z1 = fft_fftw(x, n, nbatch);
        std::cout << "CPU/GPU rel diff:  " << compare(z1, z2) << std::endl;
    }
#endif

    // t2 is in milli seconds; tera is 1e12
    double seconds      = t2 / 1000.0;
    double flop_count   = nbatch * 5.0 * n * log(double(n)) / log(2.0);
    double tflops       = flop_count / seconds / 1e12;
    double total_tflops = 0.0;
    double max_seconds  = 0.0;
#ifdef USE_MPI
    MPI_Reduce(&seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tflops, &total_tflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    max_seconds  = seconds;
    total_tflops = tflops;
#endif

    if(echo)
    {
        std::cout << "MAX TIME:         " << max_seconds << " s" << std::endl;
        std::cout << "SUM TFLOPS:       " << total_tflops << std::endl;
    }
}

int main(int argc, char* argv[])
{
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    // std::cout << "Usage: rocfft-tflops [LENGTH=512] [NBATCH=] [SINGLE=0]" << std::endl
    //           << std::endl;

    int length = 512;
    int nbatch = 500000;
    int single = 0;
    if(argc > 1)
        length = std::stoi(argv[1]);
    if(argc > 2)
        nbatch = std::stoi(argv[2]);
    if(argc > 3)
        single = std::stoi(argv[3]);

    if(single)
        test1d<hipComplex>(length, nbatch);
    else
        test1d<hipDoubleComplex>(length, nbatch);

#ifdef USE_MPI
    MPI_Finalize();
#endif
}
