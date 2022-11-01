# FFT benchmarks

## cuFFT

```sh
/usr/local/cuda-11.6/bin/nvcc -L /usr/local/cuda-11.6/targets/x86_64-linux/lib/ -lcudart -lcufft -o cufft_benchmark cufft_benchmark.cu
./cufft_benchmark
```

## hipFFT

```sh
hipcc -L/opt/rocm/lib/ -lhipfft -o hipfft_benchmark hipfft_benchmark.cpp
./hipfft_benchmark
```

## VkFFT on CUDA

```sh
/usr/local/cuda-11.6/bin/nvcc -L /usr/local/cuda-11.6/targets/x86_64-linux/lib/ -lnvrtc -lnvrtc-builtins -lcuda -lcudart -o vkfft_benchmark vkfft_benchmark.cu
./vkfft_benchmark
```

## VkFFT on HIP

```sh
hipcc -o vkfft_benchmark vkfft_benchmark.cpp
./vkfft_benchmark
```
