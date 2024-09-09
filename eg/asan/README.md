https://rocm.docs.amd.com/en/docs-6.1.0/conceptual/using-gpu-sanitizer.html

export HSA_XNACK=1

$ hipcc -g --offload-arch=gfx90a:xnack+ -fsanitize=address -shared-libsan mini.hip -o mini

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib/clang/18/lib/linux/:$LD_LIBRARY_PATH
