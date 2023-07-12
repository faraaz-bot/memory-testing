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

// Hermitrizing/Complex Conjugation Kernel
__global__ void impose_expected_format_2d(const float* obuffer_transpose,
                                          float*       obuffer,
                                          size_t       ostride_transpose_x,
                                          size_t       ostride_transpose_y,
                                          size_t       ostride_x,
                                          size_t       ostride_y,
                                          size_t       length_x,
                                          size_t       length_y)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idy < length_y / 2 + 1)
    {
        if(idx < length_x / 2 + 1)
        {
            size_t obuffer_pos           = 2 * (idx * ostride_x + idy * ostride_y);
            size_t obuffer_transpose_pos = 2 * (idx * ostride_transpose_x + idy * ostride_transpose_y);

            obuffer[obuffer_pos] = obuffer_transpose[obuffer_transpose_pos];
            obuffer[obuffer_pos + 1] = obuffer_transpose[obuffer_transpose_pos + 1];
        }
        else if(idx < length_x)
        {
            size_t obuffer_pos           = 2 * (idx * ostride_x + idy * ostride_y);
            size_t obuffer_transpose_pos = 2 * ((length_x - idx) * ostride_transpose_x
                                           + (idy == 0 ? 0 : length_y - idy) * ostride_transpose_y);

            obuffer[obuffer_pos] = obuffer_transpose[obuffer_transpose_pos];
            obuffer[obuffer_pos + 1] = -obuffer_transpose[obuffer_transpose_pos + 1];
        }
    }
}
// TO DO: void impose_expected_format_3d

void transpose(std::vector<size_t>&       length,
               std::vector<size_t>&       stride,
               const std::vector<size_t>& axes)
{
    std::vector<size_t> tmp_length;
    std::vector<size_t> tmp_stride;
    for(size_t axis : axes)
    {
        tmp_length.push_back(length[axis]);
        tmp_stride.push_back(stride[axis]);
    }
    length.swap(tmp_length);
    stride.swap(tmp_stride);
}

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
            size_t idx     = xval * stride[0] + yval * stride[1];
            buffer[idx].y = 0;
        }
        // x-axes:
        for(size_t i = 1; i < length[0] / 2; i++)
        {
            size_t idx_dest     = (length[0] - i) * stride[0] + yval * stride[1];
            size_t idx_src      = i * stride[0] + yval * stride[1];
            buffer[idx_dest].x = buffer[idx_src].x;
            buffer[idx_dest].y = -buffer[idx_src].y;
        }
    }
}
// TO DO: void symmetrize_3d

// overload for real
void fill_buffer_2d(std::vector<float>&        buffer,
                    const std::vector<size_t>& length,
                    const std::vector<size_t>& stride)
{
    for(size_t i = 0; i < length[0]; ++i)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            size_t idx     = i * stride[0] + j * stride[1];
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
            size_t idx     = i * stride[0] + j * stride[1];
            buffer[idx].x = sin(i * 1 - j * 2) + cos((i + 3) * (j + 4));
            buffer[idx].y = sin(i * 4 - j * 3) + cos((i + 2) * (j + 1));
        }
    }
}
// TO DO: void fill_buffer_3d

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

            std::cout << std::left << std::setw(12) << std::setfill(' ') << ss.str()
                        << std::flush;
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
            ss << std::noshowpos << buffer[idx].x << std::showpos << buffer[idx].y << "j" << std::flush;
            
            std::cout << std::left << std::setw(21) << std::setfill(' ') << ss.str() << std::flush;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
// TO DO: void print_buffer_3d

size_t compute_ptrdiff(const std::vector<size_t>& length, const std::vector<size_t>& stride)
{
    size_t val = 1;
    for(size_t i = 0; i < length.size(); ++i)
    {
        val += (length[i] - 1) * stride[i];
    }
    return val;
}

int main()
{
    std::vector<size_t> length = {4, 5}; // {LenX, LenY}

    std::vector<size_t> ilength = length;
    std::vector<size_t> istride = {1, ilength[0]}; // {istrideX, istrideY}

    std::vector<size_t> olength = {length[0], length[1] / 2 + 1};
    std::vector<size_t> ostride = {1, olength[0]};

    std::vector<float> ibuffer(compute_ptrdiff(ilength, istride)); // real

    // initialize data
    fill_buffer_2d(ibuffer, ilength, istride);
    // print_buffer_2d(ibuffer, ilength, istride);

    // no need to transpose
    if(ilength[0] % 2 != 0 && ilength[1] % 2 == 0
       || istride[0] > istride[1] && (ilength[0] % 2 != 0 || ilength[1] % 2 == 0))
    {
        std::cout << "No need to transpose" << std::endl;
        return 0;
    }

    // transpose directions
    std::vector<size_t> axes = {1, 0};
    transpose(ilength, istride, axes);
    // print_buffer_2d(ibuffer, ilength, istride);

    // make fake real-complex transform output
    std::vector<size_t> olength_transpose = {ilength[0], ilength[1] / 2 + 1};
    std::vector<size_t> ostride_transpose = {1, olength_transpose[0]};

    size_t osize_transpose = compute_ptrdiff(olength_transpose, ostride_transpose);
    std::vector<float2> obuffer_transpose(osize_transpose); // complex

    fill_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);
    // print_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);

    symmetrize_2d(obuffer_transpose, ilength, ostride_transpose);
    // print_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);

    // transpose back
    transpose(olength_transpose, ostride_transpose, axes);
    // print_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);

    // impose expected format
    size_t              osize = compute_ptrdiff(olength, ostride);
    std::vector<float2> obuffer(osize);

    size_t obuffer_bytes           = 2 * osize * sizeof(float);
    size_t obuffer_transpose_bytes = 2 * osize_transpose * sizeof(float);

    float* d_obuffer_transpose;
    assert(hipMalloc(&d_obuffer_transpose, obuffer_transpose_bytes) == hipSuccess);
    assert(hipMemcpy(d_obuffer_transpose,
                     obuffer_transpose.data(),
                     obuffer_transpose_bytes,
                     hipMemcpyHostToDevice)
           == hipSuccess);

    float* d_obuffer;
    assert(hipMalloc(&d_obuffer, obuffer_bytes) == hipSuccess);

    hipLaunchKernelGGL(impose_expected_format_2d,
                       dim3(32, 32),
                       dim3(ceildiv(olength[1], 32), ceildiv(olength[0], 32)),
                       0,
                       0,
                       d_obuffer_transpose,
                       d_obuffer,
                       ostride_transpose[0],
                       ostride_transpose[1],
                       ostride[0],
                       ostride[1],
                       length[0],
                       length[1]);

    assert(hipMemcpy(obuffer.data(), d_obuffer, obuffer_bytes, hipMemcpyDeviceToHost)
           == hipSuccess);

    // output
    // print_buffer_2d(obuffer, olength, ostride);

    // Release device memory
    assert(hipFree(d_obuffer_transpose) == hipSuccess);
    assert(hipFree(d_obuffer) == hipSuccess);

    return 0;
}