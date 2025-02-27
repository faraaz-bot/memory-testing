// Transpose kernel optionally using LDS

// RTC really wants C linkage so we don't need to deal with C++ name
// mangling
#ifndef TRANSPOSE_RUNTIME_COMPILE
template <typename Tval>
#else
extern "C"
#endif
__global__ void __launch_bounds__(1024, 1) transpose(const Tval* __restrict__ idata,
                                                     Tval* __restrict__ odata,
                                                     const int Nx,
                                                     const int Ny,
                                                     const int tileDim,
                                                     const int padding)
{
#if USE_LDS

    extern __shared__ __align__(sizeof(Tval)) unsigned char shmem_ptr[];
    Tval* lds = reinterpret_cast<Tval*>(shmem_ptr);

    // Input indices: straight copy.
#ifdef ROW_MAJOR
    const int gipos
        = (blockIdx.x * blockDim.x + threadIdx.x) * Ny + (blockIdx.y * blockDim.y + threadIdx.y);
    const int lipos = threadIdx.x * (tileDim + padding) + threadIdx.y;
#else
    // Column-major.
    const int gipos
        = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * Nx;
    const int lipos = threadIdx.x * (tileDim + padding) + threadIdx.y  ;
#endif

    // Contiguous read
    if((blockIdx.x * blockDim.x + threadIdx.x) < Nx && (blockIdx.y * blockDim.y + threadIdx.y) < Ny)
    {
        lds[lipos] = idata[gipos];
    }

    __syncthreads();

    // Output indices
#ifdef ROW_MAJOR
    const int lopos = threadIdx.y * (tileDim + padding) + threadIdx.x;
    const int gopos
        = (blockIdx.y * blockDim.y + threadIdx.x) * Nx + (blockIdx.x * blockDim.x + threadIdx.y);
#else
    const int lopos = threadIdx.y * (tileDim + padding) + threadIdx.x;
    const int gopos
        = (blockIdx.y * blockDim.y + threadIdx.x) + (blockIdx.x * blockDim.x + threadIdx.y) * Ny;
#endif
    
    // Contiguous write
    if((blockIdx.y * blockDim.y + threadIdx.x) < Ny && (blockIdx.x * blockDim.x + threadIdx.y) < Nx)
    {
        odata[gopos] = lds[lopos];
    }
#else
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < Ny && iy < Nx)
    {
        odata[ix * Nx + iy] = idata[iy * Ny + ix];
    }
#endif
}
