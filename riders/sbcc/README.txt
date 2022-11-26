Create a build directory, change into it, then run:

  cmake -DCMAKE_CXX_COMPILER=hipcc ..

And then run 'make' to build sbcc-rider.

To enable runtime compilation, add:

  -DSBCC_RUNTIME_COMPILE=ON

to the cmake command line.

GPU architecture may be explicitly set by passing something like:

  -DCMAKE_CXX_FLAGS=--offload-arch=gfx90a

to cmake.
