#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
intT1 ceildiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

// Transpose Kernel
template <typename Tval>
__global__ void transpose_2d(const Tval*  ibuffer,
                             Tval*        ibuffer_transpose,
                             const size_t istride_x,
                             const size_t istride_y,
                             const size_t length_x,
                             const size_t length_y,
                             const size_t tile_dim)
{
    extern __shared__ __align__(sizeof(Tval)) unsigned char shmem_ptr[];
    Tval* lds = reinterpret_cast<Tval*>(shmem_ptr);

    // Input indices: straight copy
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    const size_t gipos = ix * istride_x + iy * istride_y; // istride
    const size_t lipos = threadIdx.x * tile_dim + threadIdx.y;

    // Contiguous read
    if(ix < length_x && iy < length_y)
    {
        lds[lipos] = ibuffer[gipos];
    }

    __syncthreads();

    ix = blockIdx.y * blockDim.y + threadIdx.x;
    iy = blockIdx.x * blockDim.x + threadIdx.y;

    // Output indices
    const size_t lopos = threadIdx.y * tile_dim + threadIdx.x;
    const size_t gopos = ix * length_x + iy; // row-major

    // Contiguous write
    if(ix < length_y && iy < length_x)
    {
        ibuffer_transpose[gopos] = lds[lopos];
    }
}
// TO DO: transpose_3d

// Transpose and Hermitrizing/Complex Conjugation Kernel
template <typename Tval>
__global__ void resymmetrized_transpose_2d(const Tval*  obuffer_transpose,
                                           Tval*        obuffer,
                                           const size_t ostride_x,
                                           const size_t ostride_y,
                                           const size_t length_x,
                                           const size_t length_y,
                                           const size_t tile_dim)
{
    extern __shared__ __align__(sizeof(Tval)) unsigned char shmem_ptr[];
    Tval* lds = reinterpret_cast<Tval*>(shmem_ptr);

    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Input indices: straight copy
    const size_t gipos = ix * (length_x / 2 + 1) + iy; // row-major
    const size_t lipos = threadIdx.x * tile_dim + threadIdx.y;

    // Contiguous read
    if(ix < length_y / 2 + 1 && iy < length_x / 2 + 1)
    {
        lds[lipos] = obuffer_transpose[gipos];
    }

    __syncthreads();

    ix = blockIdx.y * blockDim.y + threadIdx.x;
    iy = blockIdx.x * blockDim.x + threadIdx.y;

    // Contiguous write
    if(iy < length_y / 2 + 1)
    {
        // Use LDS
        if(ix < length_x / 2 + 1)
        {
            // Output indices
            const size_t lopos = threadIdx.y * (tile_dim) + threadIdx.x;
            const size_t gopos = ix * ostride_x + iy * ostride_y; //ostride

            obuffer[gopos] = lds[lopos];
        }

        // Don't use LDS because symmetric pairs might not be in the same block
        else if(ix < length_x)
        {
            // Output indices
            const size_t gipos
                = (iy == 0 ? 0 : length_y - iy) * (length_x / 2 + 1) + (length_x - ix); // row-major
            const size_t gopos = ix * ostride_x + iy * ostride_y; // ostride

            obuffer[gopos].x = obuffer_transpose[gipos].x;
            obuffer[gopos].y = -obuffer_transpose[gipos].y;
        }
    }
}
// TO DO: resymmetrized_transpose_3d

void symmetrize_2d(std::vector<float2>&       buffer,
                   const std::vector<size_t>& length,
                   const std::vector<size_t>& stride)
{
    std::vector<size_t> xvals = {0};
    if(length[0] % 2 == 0)
    {
        xvals.push_back(length[0] / 2);
    }
    std::vector<size_t> yvals = {0};
    if(length[1] % 2 == 0)
    {
        yvals.push_back(length[1] / 2);
    }

    for(size_t yval : yvals)
    {
        // DY/Nyquists:
        for(size_t xval : xvals)
        {
            size_t idx    = xval * stride[0] + yval * stride[1];
            buffer[idx].y = 0;
        }
        // x-axes:
        for(size_t i = 1; i < length[0] / 2; i++)
        {
            size_t idx_dest    = (length[0] - i) * stride[0] + yval * stride[1];
            size_t idx_src     = i * stride[0] + yval * stride[1];
            buffer[idx_dest].x = buffer[idx_src].x;
            buffer[idx_dest].y = -buffer[idx_src].y;
        }
    }
}
// TO DO: symmetrize_3d

// overload for real
void fill_buffer_2d(std::vector<float>&        buffer,
                    const std::vector<size_t>& length,
                    const std::vector<size_t>& stride)
{
    for(size_t i = 0; i < length[0]; ++i)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            size_t idx  = i * stride[0] + j * stride[1];
            buffer[idx] = sin(i * 1 - j * 2) + cos((i + 3) * (j + 4));
        }
    }
}
// overload for complex
void fill_buffer_2d(std::vector<float2>&       buffer,
                    const std::vector<size_t>& length,
                    const std::vector<size_t>& stride)
{
    for(size_t i = 0; i < length[0]; ++i)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            size_t idx    = i * stride[0] + j * stride[1];
            buffer[idx].x = sin(i * 1 - j * 2) + cos((i + 3) * (j + 4));
            buffer[idx].y = sin(i * 4 - j * 3) + cos((i + 2) * (j + 1));
        }
    }
}
// TO DO: fill_buffer_3d

// overload for real
void print_buffer_2d(const std::vector<float>&  buffer,
                     const std::vector<size_t>& length,
                     const std::vector<size_t>& stride)
{
    for(size_t i = 0; i < length[0]; ++i)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            size_t idx = i * stride[0] + j * stride[1];

            std::stringstream ss;
            ss << std::noshowpos << buffer[idx] << std::flush;

            std::cout << std::left << std::setw(12) << std::setfill(' ') << ss.str() << std::flush;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
// overload for complex
void print_buffer_2d(const std::vector<float2>& buffer,
                     const std::vector<size_t>& length,
                     const std::vector<size_t>& stride)
{
    for(size_t i = 0; i < length[0]; ++i)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            size_t idx = i * stride[0] + j * stride[1];

            std::stringstream ss;
            ss << std::noshowpos << buffer[idx].x << std::showpos << buffer[idx].y << "j"
               << std::flush;

            std::cout << std::left << std::setw(21) << std::setfill(' ') << ss.str() << std::flush;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
// TO DO: print_buffer_3d

size_t compute_ptrdiff(const std::vector<size_t>& length, const std::vector<size_t>& stride)
{
    size_t val = 1;
    for(size_t i = 0; i < length.size(); ++i)
    {
        val += (length[i] - 1) * stride[i];
    }
    return val;
}

void test_2d()
{
    std::vector<size_t> length  = {6, 5}; // {LenX, LenY}
    std::vector<size_t> istride = {length[1], 1};

    std::vector<size_t> olength = {length[0], length[1] / 2 + 1};
    std::vector<size_t> ostride = {olength[1], 1};

    size_t             isize = compute_ptrdiff(length, istride);
    std::vector<float> ibuffer(isize); // real

    // initialize data
    fill_buffer_2d(ibuffer, length, istride);
    // print_buffer_2d(ibuffer, length, istride);

    // no need to transpose
    if(length[0] % 2 != 0 && length[1] % 2 == 0
       || istride[0] > istride[1] && (length[0] % 2 != 0 || length[1] % 2 == 0))
    {
        std::cout << "No need to transpose" << std::endl;
        return;
    }

    // transpose directions
    std::vector<size_t> length_transpose  = {length[1], length[0]};
    std::vector<size_t> istride_transpose = {length_transpose[1], 1};

    size_t             isize_transpose = compute_ptrdiff(length_transpose, istride_transpose);
    std::vector<float> ibuffer_transpose(isize_transpose);

    size_t ibuffer_bytes           = isize * sizeof(float);
    size_t ibuffer_transpose_bytes = isize_transpose * sizeof(float);

    void* d_ibuffer;
    assert(hipMalloc(&d_ibuffer, ibuffer_bytes) == hipSuccess);
    assert(hipMemcpy(d_ibuffer, ibuffer.data(), ibuffer_bytes, hipMemcpyHostToDevice)
           == hipSuccess);

    void* d_ibuffer_transpose;
    assert(hipMalloc(&d_ibuffer_transpose, ibuffer_transpose_bytes) == hipSuccess);

    hipLaunchKernelGGL(transpose_2d,
                       dim3(32, 32),
                       dim3(ceildiv(length[1], 32), ceildiv(length[0], 32)),
                       32 * 32 * sizeof(float),
                       0,
                       (float*)d_ibuffer,
                       (float*)d_ibuffer_transpose,
                       istride[0],
                       istride[1],
                       length[0],
                       length[1],
                       32);

    assert(hipMemcpy(ibuffer_transpose.data(),
                     d_ibuffer_transpose,
                     ibuffer_transpose_bytes,
                     hipMemcpyDeviceToHost)
           == hipSuccess);

    assert(hipFree(d_ibuffer) == hipSuccess);
    assert(hipFree(d_ibuffer_transpose) == hipSuccess);

    // print_buffer_2d(ibuffer_transpose, length_transpose, istride_transpose);

    // make fake real-complex transform output
    std::vector<size_t> olength_transpose = {length_transpose[0], length_transpose[1] / 2 + 1};
    std::vector<size_t> ostride_transpose = {olength_transpose[1], 1};

    size_t              osize_transpose = compute_ptrdiff(olength_transpose, ostride_transpose);
    std::vector<float2> obuffer_transpose(osize_transpose);

    fill_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);
    // print_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);

    symmetrize_2d(obuffer_transpose, length, ostride_transpose);
    // print_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);

    // transpose back and impose expected format
    size_t              osize = compute_ptrdiff(olength, ostride);
    std::vector<float2> obuffer(osize);

    size_t obuffer_transpose_bytes = osize_transpose * sizeof(std::complex<float>);
    size_t obuffer_bytes           = osize * sizeof(std::complex<float>);

    void* d_obuffer_transpose;
    assert(hipMalloc(&d_obuffer_transpose, obuffer_transpose_bytes) == hipSuccess);
    assert(hipMemcpy(d_obuffer_transpose,
                     obuffer_transpose.data(),
                     obuffer_transpose_bytes,
                     hipMemcpyHostToDevice)
           == hipSuccess);

    void* d_obuffer;
    assert(hipMalloc(&d_obuffer, obuffer_bytes) == hipSuccess);

    hipLaunchKernelGGL(resymmetrized_transpose_2d,
                       dim3(32, 32),
                       dim3(ceildiv(olength[1], 32), ceildiv(olength[0], 32)),
                       32 * 32 * sizeof(std::complex<float>),
                       0,
                       (float2*)d_obuffer_transpose,
                       (float2*)d_obuffer,
                       ostride[0],
                       ostride[1],
                       length[0],
                       length[1],
                       32);

    assert(hipMemcpy(obuffer.data(), d_obuffer, obuffer_bytes, hipMemcpyDeviceToHost)
           == hipSuccess);

    assert(hipFree(d_obuffer_transpose) == hipSuccess);
    assert(hipFree(d_obuffer) == hipSuccess);

    // output
    // print_buffer_2d(obuffer, olength, ostride);
}
// TO DO: test_3d

int main()
{
    test_2d();

    return 0;
}