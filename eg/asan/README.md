https://rocm.docs.amd.com/en/docs-6.1.0/conceptual/using-gpu-sanitizer.html

# Install needed packages:
sudo apt install hsa-rocr-asan hip-runtime-amd-asan hsakmt-roct-asan openmp-extras-asan rocm-hip-runtime-asan rocm-language-runtime-asan rocm-smi-lib-asan

# Set up environment and compile:
export HSA_XNACK=1
export LIBRARTY_PATH=/opt/rocm/lib/llvm/lib/asan:/opt/rocm/lib/asan

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
