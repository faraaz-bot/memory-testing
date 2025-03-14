#include <benchmark/benchmark.h>

#include "mem-bench.hpp"

int main(int argc, char* argv[])
{
    CLI::App app{"Memcpy bench"};

    size_t N;
    size_t ngpus;
    int verbose;
    app.add_option("-n, --length", N, "Length of input square matrix")->default_val(8U);
    app.add_option("-g, --ngpus", ngpus, "Number of gpus")->default_val(4U);
    app.add_option("-V, --verbose", verbose, "Adjust output verbosity level")->default_val(0);

    // TODO option: precision, input generation (host, dev, random, sequence?), which benchmark(s) to run
    // , output format options

    app.allow_extras();
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    std::cout << "Comparing on " << N << " x " << N << " size matrix, across " << ngpus << " gpus.\n";

    // Generate random input
    // Can consider adding in option to use rocRAND for faster device generation
    std::random_device                    rd;
    std::mt19937                          m_engine(rd()); // Mersenne Twister, rd as seed
    std::uniform_real_distribution<float> dist{-0.5, 0.5};

    // std::vector<float> input(N*N);
// # pragma omp parallel for
    // for(size_t i = 0; i < N*N; ++i)
    //     input[i] = dist(m_engine);
    
    // For debugging, [1,2,3,..N*N]
    std::vector<float> input(N*N);
    for(size_t i = 0; i < N*N; i++)
        input[i] = i;
    
    std::cout << "Input Matrix:\n";
    print_host_2d(N, N, input);

    std::vector<float> reference_matrix(N*N);
    host_transpose(N, input, reference_matrix);
    std::cout << "Host Transposed Matrix:\n";
    print_host_2d(N,N,reference_matrix);

    // Split input and transfer it
    // Assume inputs are evenly divisible :)
    std::vector<float*> gpubufs_input(ngpus);
    std::vector<float*> gpubufs_output(ngpus);
    std::vector<hipStream_t> streams(ngpus);

    const size_t buf_height = N / ngpus;
    const size_t buf_size = N * buf_height; // Number of elements in buf
    const size_t pitch_bytes = N * sizeof(float); // Size of a column in bytes incl. padding (which is 0)
    
    std::cout << "buf_height = " << buf_height << "\nbuf_size = " << buf_size << "\npitch_bytes = " << pitch_bytes << std::endl;

    // Allocate and init bufs, streams
    for(size_t i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));

        HIP_CHECK(hipMalloc(&gpubufs_input[i], sizeof(float) * buf_size)); 
        HIP_CHECK(hipMemcpy(gpubufs_input[i], input.data() + i * buf_size, buf_size * sizeof(float), hipMemcpyHostToDevice));
        // HIP_CHECK(hipMemcpy2D(gpubufs_input[i], pitch_bytes, input.data() + i * buf_size, pitch_bytes, N, buf_height, hipMemcpyHostToDevice));
        
        HIP_CHECK(hipMalloc(&gpubufs_output[i], sizeof(float) * buf_size));
        HIP_CHECK(hipMemset(gpubufs_output[i], 0, sizeof(float) * buf_size));
        std::cout << "Input GPU Buffer " << i << ":\n";
        print<<<1,1>>>(buf_size, gpubufs_input[i]);
        print2d<<<1,1>>>(N, buf_height, gpubufs_input[i]);

        // Assign streams to current gpu
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    // Enable peer to peer memory access between GPUs
    for(size_t i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        for(size_t j = 0; j < ngpus; j++)
        {
            int can_access_peer;
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, i, j));
            if(can_access_peer)
                HIP_CHECK(hipDeviceEnablePeerAccess(j, 0));
        }
    }

    // -- Run stuff --

    std::vector<float> h_assembled_output(N*N);
    run_memcpy(N, gpubufs_input, gpubufs_output);
    assemble_output_to_host(N, gpubufs_output, h_assembled_output.data());
    bool res = is_same_matrix(N, reference_matrix, h_assembled_output);
    std::cout << "Are two matrices equal? " << res << "\nOutput Assembled on Host:\n"; // Currently should not, due to lack of local transpose!
    print_host_2d(N,N,h_assembled_output);

    // Implement cleanup -> fill/memset existing bufs with 0?

    // Copy kernel
    // MPI alltoall
    // RCCL alltoall
    
    // Free up buffers, streams
    for(auto i = 0; i < ngpus; i++){
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(gpubufs_input[i]));
        HIP_CHECK(hipFree(gpubufs_output[i]));
        HIP_CHECK(hipStreamDestroy(streams[i])); 
    }

    // Disabling peer access
    // for(size_t i = 0; i < ngpus; i++)
    // {
    //     HIP_CHECK(hipSetDevice(i));
    //     for(size_t j = 0; j < ngpus; j++)
    //     {
    //         int can_access_peer;
    //         HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, i, j));
    //         if(can_access_peer)
    //             hipDeviceDisablePeerAccess(j);
    //     }
    // }
}

