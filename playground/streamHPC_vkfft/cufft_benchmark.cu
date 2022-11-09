#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <locale>
#include <codecvt>
#include <string>

#include <cuda_runtime.h>

#include <cufft.h>

#define CUDA_CHECK(condition)         \
  {                                  \
    cudaError_t error = condition;    \
    if(error != cudaSuccess){         \
        std::cout << "CUDA error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

static void handleCufftError(cufftResult_t status, const char* msg)
{
    if (status != CUFFT_SUCCESS)
    {
        std::cout << "cuFFT error: " << status << " line: " << __LINE__ << std::endl; \
        exit(status);
    }
}

constexpr unsigned int XX = 0;
constexpr unsigned int YY = 1;
constexpr unsigned int ZZ = 2;

double run_benchmark(int nx, int ny, int nz, cudaStream_t stream)
{
    std::vector<float> input(nx * ny * nz);
    for (int i = 0; i < nx * ny * nz; i++)
    {
        input[i] = float(i % 123 - 70);
    }

    float * realGrid;
    float2 * complexGrid;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&realGrid), nx * ny * nz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&complexGrid), nx * ny * (nz / 2 + 1) * sizeof(float2)));
    CUDA_CHECK(
        cudaMemcpy(
            realGrid, input.data(),
            nx * ny * nz * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int realGridSize[3] = { nx, ny, nz };
    int realGridSizePadded[3] = { nx, ny, nz };
    int complexGridSizePadded[3] = { nx, ny, nz / 2 + 1 };
    const int complexGridSizePaddedTotal =
            complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
    const int realGridSizePaddedTotal =
            realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ];

    const int rank = 3, batch = 1;
    cufftResult_t result;
    cufftHandle   planR2C;
    cufftHandle   planC2R;

    result = cufftPlanMany(&planR2C,
                           rank,
                           realGridSize,
                           realGridSizePadded,
                           1,
                           realGridSizePaddedTotal,
                           complexGridSizePadded,
                           1,
                           complexGridSizePaddedTotal,
                           CUFFT_R2C,
                           batch);
    handleCufftError(result, "cufftPlanMany R2C plan failure");

    result = cufftPlanMany(&planC2R,
                           rank,
                           realGridSize,
                           complexGridSizePadded,
                           1,
                           complexGridSizePaddedTotal,
                           realGridSizePadded,
                           1,
                           realGridSizePaddedTotal,
                           CUFFT_C2R,
                           batch);
    handleCufftError(result, "cufftPlanMany C2R plan failure");

    result = cufftSetStream(planR2C, stream);
    handleCufftError(result, "cufftSetStream R2C failure");

    result = cufftSetStream(planC2R, stream);
    handleCufftError(result, "cufftSetStream C2R failure");

    auto run = [&]()
    {
        result = cufftExecR2C(planR2C, realGrid, complexGrid);
        handleCufftError(result, "cuFFT R2C execution failure");

        result = cufftExecC2R(planC2R, complexGrid, realGrid);
        handleCufftError(result, "cuFFT C2R execution failure");
    };

    constexpr unsigned int runs = 1000;
    constexpr unsigned int warmups = 10;

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warm-up
    for (unsigned int i = 0; i < warmups; i++)
    {
        run();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> ds;
    double avg = 0;
    for (int i = 0; i < runs; i++)
    {
        cudaEventRecord(start, stream);

        run();

        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        float ms;
        cudaEventElapsedTime(&ms, start, end);
        double us  = ms * 1e-3;
        ds.push_back(us);
        avg += us;
    }
    std::sort(ds.begin(), ds.end());
    avg /= runs;
    // printf(" ");
    // printf("median: %8.3f\n", ds[runs / 2] * 1e6);
    // printf("avg:    %8.3f\n", avg * 1e6);
    // printf("min:    %8.3f\n", ds[0] * 1e6);
    // printf("max:    %8.3f\n", ds[runs - 1] * 1e6);

    result = cufftDestroy(planR2C);
    handleCufftError(result, "cufftDestroy R2C failure");
    result = cufftDestroy(planC2R);
    handleCufftError(result, "cufftDestroy C2R failure");

    CUDA_CHECK(cudaFree(realGrid));
    CUDA_CHECK(cudaFree(complexGrid));

    return ds[runs / 2] * 1e6;
}

std::array<std::string, 2> get_radix_string(unsigned int n)
{
    std::string s;
    constexpr unsigned int rs[] = { 2, 3, 5, 7, 11, 13 };
    for (auto r : rs)
    {
        int k = 0;
        while (n % r == 0)
        {
            n /= r;
            k++;
        }
        if (k != 0)
        {
            if (!s.empty()) s += "*";
            s += std::to_string(r) + "^" + std::to_string(k);
        }
    }
    return std::array<std::string, 2>{ s, n != 1 ? std::to_string(n) : std::string("") };
}

int main(int argc, char *argv[])
{
    cudaDeviceProp dev_prop;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, device));
    std::cout << "Device: " << dev_prop.name << std::endl;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<std::array<std::string, 11>> csv;
    auto run = [&](const std::string& name, unsigned int nx, unsigned int ny, unsigned int nz)
    {
        auto ax = get_radix_string(nx);
        auto ay = get_radix_string(ny);
        auto az = get_radix_string(nz);
        double d = run_benchmark(nx, ny, nz, stream);
        printf("%-15s %3d %3d %3d   %-15s%4s   %-15s%4s   %-15s%4s   %6.0f\n",
            name.c_str(),
            nx, ny, nz,
            ax[0].c_str(), ax[1].c_str(), ay[0].c_str(), ay[1].c_str(), az[0].c_str(), az[1].c_str(),
            d
        );
        csv.push_back(std::array<std::string, 11>{
            name,
            std::to_string(nx), std::to_string(ny), std::to_string(nz),
            ax[0], ax[1], ay[0], ay[1], az[0], az[1],
            std::to_string(int(d))
        });
    };

    run("aqp", 84, 84, 72);
    run("eag1", 128, 128, 120);
    run("adh", 100, 100, 100);
    run("cellulose ini", 280, 128, 128);
    run("cellulose opt", 256, 128, 128);
    run("stmv ini", 160, 160, 160);
    run("stmv opt", 144, 144, 144);

    for (unsigned int n = 64; n <= 128; n += 1)
    {
        run(std::to_string(n), n, n, n);
    }

    printf("\n");
    for (auto c : csv)
    {
        printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", c[0].c_str(), c[1].c_str(), c[2].c_str(), c[3].c_str(), c[4].c_str(), c[5].c_str(), c[6].c_str(), c[7].c_str(), c[8].c_str(), c[9].c_str(), c[10].c_str());
    }
    printf("\n");
    for (auto c : csv)
    {
        printf("%s\n", c[10].c_str());
    }
    printf("\n");
}
