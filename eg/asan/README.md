https://rocm.docs.amd.com/en/docs-6.1.0/conceptual/using-gpu-sanitizer.html

export HSA_XNACK=1

$ hipcc -g --offload-arch=gfx90a:xnack+ -fsanitize=address -shared-libsan asan.cpp -o mini
amdclang++ -g --offload-arch=gfx90a:xnack+ -fsanitize=address -shared-libsan asan.cpp

rm -rf * && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j1 VERBOSE=1

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib/clang/18/lib/linux/:$LD_LIBRARY_PATH
