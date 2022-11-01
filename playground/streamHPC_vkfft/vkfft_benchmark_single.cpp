#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <locale>
#include <codecvt>
#include <string>

#include <hip/hip_runtime.h>

#define VKFFT_BACKEND 2
#include "vkFFT.h"

#define HIP_CHECK(condition)                                                                       \
  {                                                                                                \
    hipError_t error = condition;                                                                  \
    if (error != hipSuccess) {                                                                     \
      std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl;                   \
      exit(error);                                                                                 \
    }                                                                                              \
  }

constexpr unsigned int XX = 0;
constexpr unsigned int YY = 1;
constexpr unsigned int ZZ = 2;

double run_benchmark(unsigned int nx, unsigned int ny, unsigned int nz, hipStream_t stream,
                     bool round_trip = true) {
  const unsigned int realGridSize[3] = {nx, ny, nz};
  const unsigned int realGridSizePadded[3] = {nx, ny, nz};
  const unsigned int complexGridSizePadded[3] = {nx, ny, nz / 2 + 1};

  uint64_t bufferSize = complexGridSizePadded[XX] * complexGridSizePadded[YY] *
      complexGridSizePadded[ZZ] * sizeof(float2);
  uint64_t inputBufferSize =
      realGridSizePadded[XX] * realGridSizePadded[YY] * realGridSizePadded[ZZ] * sizeof(float);

  std::vector<float> input(realGridSizePadded[XX] * realGridSizePadded[YY] *
                           realGridSizePadded[ZZ]);
  for (size_t i = 0; i < input.size(); i++) {
    input[i] = float(i % 123) - 70;
  }

  float* realGrid;
  float2* complexGrid;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&realGrid), inputBufferSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&complexGrid), bufferSize));
  HIP_CHECK(hipMemcpy(realGrid, input.data(), inputBufferSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());

  VkFFTConfiguration configuration = {};
  VkFFTApplication appR2C = {};

  configuration.FFTdim = 3;
  configuration.size[0] = realGridSize[ZZ];
  configuration.size[1] = realGridSize[YY];
  configuration.size[2] = realGridSize[XX];

  configuration.performR2C = 1;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  configuration.device = &device;
  configuration.stream = &stream;
  configuration.num_streams = 1;

  // configuration.useLUT = 1;
  // configuration.disableMergeSequencesR2C = 1;

  configuration.bufferSize = &bufferSize;
  configuration.bufferStride[0] = complexGridSizePadded[ZZ];
  configuration.bufferStride[1] = complexGridSizePadded[ZZ] * complexGridSizePadded[YY];
  configuration.bufferStride[2] =
      complexGridSizePadded[ZZ] * complexGridSizePadded[YY] * complexGridSizePadded[XX];
  configuration.buffer = (void**)&complexGrid;

  configuration.isInputFormatted = 1;
  configuration.inverseReturnToInputBuffer = 1;
  configuration.inputBufferSize = &inputBufferSize;
  configuration.inputBufferStride[0] = realGridSizePadded[ZZ];
  configuration.inputBufferStride[1] = realGridSizePadded[ZZ] * realGridSizePadded[YY];
  configuration.inputBufferStride[2] =
      realGridSizePadded[ZZ] * realGridSizePadded[YY] * realGridSizePadded[XX];
  configuration.inputBuffer = (void**)&realGrid;

  VkFFTResult resFFT;
  resFFT = initializeVkFFT(&appR2C, configuration);
  if (resFFT != VKFFT_SUCCESS) {
    printf("VkFFT error: %d\n", resFFT);
    exit(-1);
  }

  auto run = [&]() {
    VkFFTResult resFFT;
    resFFT = VkFFTAppend(&appR2C, -1, NULL);
    if (resFFT != VKFFT_SUCCESS) {
      printf("VkFFT error: %d\n", resFFT);
      exit(-1);
    }
    if (round_trip) {
      resFFT = VkFFTAppend(&appR2C, +1, NULL);
      if (resFFT != VKFFT_SUCCESS) {
        printf("VkFFT error: %d\n", resFFT);
        exit(-1);
      }
    }
  };

  unsigned int runs = 1000;
  unsigned int batch = 10;
  unsigned int warmups = 10;

  if (getenv("ISA")) {
    runs = 1;
    batch = 1;
    warmups = 0;
  }

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

  deleteVkFFT(&appR2C);

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

  // run("aqp", 84, 84, 72);
  // run("eag1", 128, 128, 120);
  // run("adh", 100, 100, 100);
  // run("cellulose ini", 280, 128, 128);
  // run("cellulose opt", 256, 128, 128);
  // run("stmv ini", 160, 160, 160);
  // run("stmv opt", 144, 144, 144);

  // for (unsigned int n = 64; n <= 128; n += 1) {
  //   run(std::to_string(n), n, n, n);
  // }

  // if (!getenv("ISA")) {
  //   // for (unsigned int n = 76; n <= 96; n += 1)
  //   // {
  //   //     for (unsigned int m = 64; m <= 84; m += 1)
  //   //     {
  //   //         if (get_radix_string(n)[1] == "" && get_radix_string(m)[1] == "")
  //   //         {
  //   //             run(std::to_string(n) + " " + std::to_string(m), n, n, m);
  //   //         }
  //   //     }
  //   // }

  //   printf("\n");
  //   for (auto c : csv) {
  //     printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", c[0].c_str(), c[1].c_str(), c[2].c_str(),
  //            c[3].c_str(), c[4].c_str(), c[5].c_str(), c[6].c_str(), c[7].c_str(), c[8].c_str(),
  //            c[9].c_str(), c[10].c_str());
  //   }
  //   printf("\n");
  //   for (auto c : csv) {
  //     printf("%s\n", c[10].c_str());
  //   }
  //   printf("\n");
  // }
}
