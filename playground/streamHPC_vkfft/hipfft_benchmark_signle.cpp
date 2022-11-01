#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <locale>
#include <codecvt>
#include <string>

#include <hip/hip_runtime.h>

#include <hipfft.h>

// hipcc -L/opt/rocm/lib/ -lhipfft -o hipfft_benchmark hipfft_benchmark.cpp && ./hipfft_benchmark

#define HIP_CHECK(condition)                                                                       \
  {                                                                                                \
    hipError_t error = condition;                                                                  \
    if (error != hipSuccess) {                                                                     \
      std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl;                   \
      exit(error);                                                                                 \
    }                                                                                              \
  }

#define handleHipfftError(status, msg)                                                             \
  {                                                                                                \
    if (status != HIPFFT_SUCCESS) {                                                                \
      std::cout << "hipFFT error: " << status << " line: " << __LINE__ << std::endl;               \
      exit(status);                                                                                \
    }                                                                                              \
  }

constexpr unsigned int XX = 0;
constexpr unsigned int YY = 1;
constexpr unsigned int ZZ = 2;

double run_benchmark(int nx, int ny, int nz, hipStream_t stream, bool round_trip = true) {
  size_t bufferSize = nx * ny * (nz / 2 + 1) * sizeof(float2);
  size_t inputBufferSize = nx * ny * nz * sizeof(float);

  std::vector<float> input(nx * ny * nz);
  for (size_t i = 0; i < input.size(); i++) {
    input[i] = float(i % 123) - 70;
  }

  float* realGrid;
  float2* complexGrid;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&realGrid), inputBufferSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&complexGrid), bufferSize));
  HIP_CHECK(hipMemcpy(realGrid, input.data(), inputBufferSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());

  int realGridSize[3] = {nx, ny, nz};
  int realGridSizePadded[3] = {nx, ny, nz};
  int complexGridSizePadded[3] = {nx, ny, nz / 2 + 1};
  const int complexGridSizePaddedTotal =
      complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ];
  const int realGridSizePaddedTotal =
      realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ];

  hipfftResult_t result;
  hipfftHandle planR2C;
  hipfftHandle planC2R;

  result = hipfftPlanMany(&planR2C, 3, realGridSize, realGridSizePadded, 1, realGridSizePaddedTotal,
                          complexGridSizePadded, 1, complexGridSizePaddedTotal, HIPFFT_R2C, 1);
  handleHipfftError(result, "hipfftPlanMany R2C plan failure");

  result = hipfftPlanMany(&planC2R, 3, realGridSize, complexGridSizePadded, 1,
                          complexGridSizePaddedTotal, realGridSizePadded, 1,
                          realGridSizePaddedTotal, HIPFFT_C2R, 1);
  handleHipfftError(result, "hipfftPlanMany C2R plan failure");

  result = hipfftSetStream(planR2C, stream);
  handleHipfftError(result, "hipfftSetStream R2C failure");

  result = hipfftSetStream(planC2R, stream);
  handleHipfftError(result, "hipfftSetStream C2R failure");

  auto run = [&]() {
    result = hipfftExecR2C(planR2C, realGrid, complexGrid);
    handleHipfftError(result, "hipFFT R2C exehiption failure");

    if (round_trip) {
      result = hipfftExecC2R(planC2R, complexGrid, realGrid);
      handleHipfftError(result, "hipFFT C2R exehiption failure");
    }
  };

  unsigned int runs = 1000;
  unsigned int batch = 10;
  unsigned int warmups = 10;

  hipEvent_t start, end;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&end));
  HIP_CHECK(hipDeviceSynchronize());

  // Warm-up
  for (unsigned int i = 0; i < warmups; i++) {
    run();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<double> ds;
  double avg = 0;
  for (int i = 0; i < runs; i += batch) {
    hipEventRecord(start, stream);

    for (int j = 0; j < batch; j++) {
      run();
    }

    hipEventRecord(end, stream);
    hipEventSynchronize(end);
    float ms;
    hipEventElapsedTime(&ms, start, end);
    double us = (ms * 1e-3) / batch;
    ds.push_back(us);
    avg += us;
  }
  std::sort(ds.begin(), ds.end());
  avg /= ds.size();
  // printf(" ");
  // printf("median: %8.3f\n", ds[runs / 2] * 1e6);
  // printf("avg:    %8.3f\n", avg * 1e6);
  // printf("min:    %8.3f\n", ds[0] * 1e6);
  // printf("max:    %8.3f\n", ds[runs - 1] * 1e6);

  // Check correctness
  if (round_trip) {
    HIP_CHECK(hipMemcpy(realGrid, input.data(), inputBufferSize, hipMemcpyHostToDevice));
    run();
    std::vector<float> output(realGridSizePadded[XX] * realGridSizePadded[YY] *
                              realGridSizePadded[ZZ]);
    HIP_CHECK(hipMemcpy(output.data(), realGrid, inputBufferSize, hipMemcpyDeviceToHost));
    // Normalize
    for (size_t i = 0; i < output.size(); i++) {
      output[i] /= (nx * ny * nz);
    }
    float maxDiff = 0;
    size_t maxDiffI = 0;
    for (size_t i = 0; i < output.size(); i++) {
      const float diff = std::abs(input[i] - output[i]);
      if (diff > maxDiff) {
        maxDiff = diff;
        maxDiffI = i;
      }
    }
    if (maxDiff > 1e-3f) {
      printf("!!! %e %6zu %+e %+e\n", maxDiff, maxDiffI, input[maxDiffI], output[maxDiffI]);
    }
  }

  result = hipfftDestroy(planR2C);
  handleHipfftError(result, "hipfftDestroy R2C failure");
  result = hipfftDestroy(planC2R);
  handleHipfftError(result, "hipfftDestroy C2R failure");

  HIP_CHECK(hipFree(realGrid));
  HIP_CHECK(hipFree(complexGrid));

  return ds[ds.size() / 2] * 1e6;
}

std::array<std::string, 2> get_radix_string(unsigned int n) {
  std::string s;
  constexpr unsigned int rs[] = {2, 3, 5, 7, 11, 13};
  for (auto r : rs) {
    int k = 0;
    while (n % r == 0) {
      n /= r;
      k++;
    }
    if (k != 0) {
      if (!s.empty()) s += "*";
      s += std::to_string(r) + "^" + std::to_string(k);
    }
  }
  return std::array<std::string, 2>{s, n != 1 ? std::to_string(n) : std::string("")};
}

int main(int argc, char* argv[]) {
  hipDeviceProp_t dev_prop;
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&dev_prop, device));
  std::cout << "Device: " << dev_prop.name << std::endl;

  int x = std::stoi(argv[1]);
  int y = std::stoi(argv[2]);
  int z = std::stoi(argv[3]);
  int round_trip = std::stoi(argv[4]);

  //   std::cin >> x >> y >> z;
  std::cout << " 3D FFTs of size " << x << "x" << y << "x" << z << std::endl;
  std::cout << " round_trip " << round_trip << std::endl;


  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  std::vector<std::array<std::string, 11>> csv;
  auto run = [&](const std::string& name, unsigned int nx, unsigned int ny, unsigned int nz,
                 bool round_trip) {
    auto ax = get_radix_string(nx);
    auto ay = get_radix_string(ny);
    auto az = get_radix_string(nz);
    double d = run_benchmark(nx, ny, nz, stream, round_trip);
    printf("%-15s %3d %3d %3d   %-15s%4s   %-15s%4s   %-15s%4s   %6.0f\n", name.c_str(), nx, ny, nz,
           ax[0].c_str(), ax[1].c_str(), ay[0].c_str(), ay[1].c_str(), az[0].c_str(), az[1].c_str(),
           d);
    csv.push_back(std::array<std::string, 11>{name, std::to_string(nx), std::to_string(ny),
                                              std::to_string(nz), ax[0], ax[1], ay[0], ay[1], az[0],
                                              az[1], std::to_string(int(d))});
  };

  run("test", x, y, z, round_trip);

  //   run("aqp", 84, 84, 72);
  //   run("eag1", 128, 128, 120);
  //   run("adh", 100, 100, 100);
  //   run("cellulose ini", 280, 128, 128);
  //   run("cellulose opt", 256, 128, 128);
  //   run("stmv ini", 160, 160, 160);
  //   run("stmv opt", 144, 144, 144);

  //   for (unsigned int n = 64; n <= 128; n += 1) {
  //     // if (n == 98) // It fails for unknown reason
  //     //{
  //     //     run("failed " + std::to_string(n), 100, 100, 100);
  //     // }
  //     // else
  //     { run(std::to_string(n), n, n, n); }
  //   }

  printf("\n");
  //   for (auto c : csv) {
  //     printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", c[0].c_str(), c[1].c_str(), c[2].c_str(),
  //            c[3].c_str(), c[4].c_str(), c[5].c_str(), c[6].c_str(), c[7].c_str(), c[8].c_str(),
  //            c[9].c_str(), c[10].c_str());
  //   }
  //   printf("\n");
  //   for (auto c : csv) {
  //     printf("%s\n", c[10].c_str());
  //   }
  printf("\n");
}
