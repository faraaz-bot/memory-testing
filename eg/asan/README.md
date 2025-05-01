https://rocm.docs.amd.com/en/docs-6.1.0/conceptual/using-gpu-sanitizer.html

# Install needed packages:
sudo apt install hsa-rocr-asan hip-runtime-amd-asan openmp-extras-asan rocm-hip-runtime-asan rocm-language-runtime-asan rocm-smi-lib-asan

# Set up environment and compile:
export HSA_XNACK=1
export LIBRARY_PATH=/opt/rocm/lib/llvm/lib/asan:/opt/rocm/lib/asan

cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=amdclang++ .. && make

hipcc -g --offload-arch=gfx90a:xnack+ -fsanitize=address -shared-libsan ../asan.cpp -o mini -lboost_program_options


export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib/clang/18/lib/linux/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/lib/asan/:$LD_LIBRARY_PATH


# Test address sanitizer:

Insufficient allocation on the device:
./aotasan --m 31

Out-of-bounds memory access in the GPU kernel:
./aotasan --m 31 --c 31

Out-of-bounds memory access when copying back to the host:
./aotasan --c 31



I'm getting an error at the end of execution with 6.2.0 on MI250X:

AddressSanitizer: CHECK failed: sanitizer_allocator_device.h:214 "((h)) != ((nullptr))" (0x0, 0x0) (tid=2031078)
Tracer caught signal 11: addr=0x82000 pc=0x7f1eb5e2a83e sp=0x7f1ea8c22060
==2031072==LeakSanitizer has encountered a fatal error.
==2031072==HINT: For debugging, try setting environment variable LSAN_OPTIONS=verbosity=1:log_threads=1
==2031072==HINT: LeakSanitizer does not work under ptrace (strace, gdb, etc)


This can be removed by running

ASAN_OPTIONS=detect_leaks=0 
