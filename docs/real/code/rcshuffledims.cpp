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
__global__ void impose_expected_format_2d(const float* obuffer_transpose_real,
                                          const float* obuffer_transpose_imag,
                                          float*       obuffer_real,
                                          float*       obuffer_imag,
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
            size_t obuffer_pos           = idx * ostride_x + idy * ostride_y;
            size_t obuffer_transpose_pos = idx * ostride_transpose_x + idy * ostride_transpose_y;

            obuffer_real[obuffer_pos] = obuffer_transpose_real[obuffer_transpose_pos];
            obuffer_imag[obuffer_pos] = obuffer_transpose_imag[obuffer_transpose_pos];
        }
        else if(idx < length_x)
        {
            size_t obuffer_pos           = idx * ostride_x + idy * ostride_y;
            size_t obuffer_transpose_pos = (length_x - idx) * ostride_transpose_x
                                           + (idy == 0 ? 0 : length_y - idy) * ostride_transpose_y;

            obuffer_real[obuffer_pos] = obuffer_transpose_real[obuffer_transpose_pos];
            obuffer_imag[obuffer_pos] = -obuffer_transpose_imag[obuffer_transpose_pos];
        }
    }
}
// TO DO: void impose_expected_format_3d

// Should this be a kernel?
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

void symmetrize_2d(std::vector<std::vector<float>>& buffer,
                   const std::vector<size_t>&       length,
                   const std::vector<size_t>&       stride)
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
            buffer[1][idx] = 0;
        }
        // x-axes:
        for(size_t i = 1; i < length[0] / 2; i++)
        {
            size_t idx_dest     = (length[0] - i) * stride[0] + yval * stride[1];
            size_t idx_src      = i * stride[0] + yval * stride[1];
            buffer[0][idx_dest] = buffer[0][idx_src];
            buffer[1][idx_dest] = -buffer[1][idx_src];
        }
    }
}
// TO DO: void symmetrize_3d

void fill_buffer_2d(std::vector<std::vector<float>>& buffer,
                    const std::vector<size_t>&       length,
                    const std::vector<size_t>&       stride)
{
    if(buffer.size() == 1) // real input
    {
        for(size_t i = 0; i < length[0]; ++i)
        {
            for(size_t j = 0; j < length[1]; ++j)
            {
                size_t idx     = i * stride[0] + j * stride[1];
                buffer[0][idx] = sin(i * 1 - j * 2) + cos((i + 3) * (j + 4));
            }
        }
    }
    else // complex output
    {
        for(size_t i = 0; i < length[0]; ++i)
        {
            for(size_t j = 0; j < length[1]; ++j)
            {
                size_t idx     = i * stride[0] + j * stride[1];
                buffer[0][idx] = sin(i * 1 - j * 2) + cos((i + 3) * (j + 4)); // real part
                buffer[1][idx] = sin(i * 4 - j * 3) + cos((i + 2) * (j + 1)); // complex part
            }
        }
    }
}
// TO DO: void fill_buffer_3d

void print_buffer_2d(const std::vector<std::vector<float>>& buffer,
                     const std::vector<size_t>&             length,
                     const std::vector<size_t>&             stride)
{
    if(buffer.size() == 1) // real
    {
        for(size_t i = 0; i < length[0]; ++i)
        {
            for(size_t j = 0; j < length[1]; ++j)
            {
                size_t idx = i * stride[0] + j * stride[1];

                std::stringstream ss;
                ss << std::noshowpos << buffer[0][idx] << std::flush;

                std::cout << std::left << std::setw(12) << std::setfill(' ') << ss.str()
                          << std::flush;
            }
            std::cout << std::endl;
        }
    }
    else // complex
    {
        for(size_t i = 0; i < length[0]; ++i)
        {
            for(size_t j = 0; j < length[1]; ++j)
            {
                size_t idx = i * stride[0] + j * stride[1];

                std::stringstream ss;
                ss << std::noshowpos << buffer[0][idx] << std::showpos << buffer[1][idx] << "j"
                   << std::flush;

                std::cout << std::left << std::setw(21) << std::setfill(' ') << ss.str()
                          << std::flush;
            }
            std::cout << std::endl;
        }
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

    std::vector<std::vector<float>> ibuffer(1); // real
    ibuffer[0].resize(compute_ptrdiff(ilength, istride));

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

    std::vector<std::vector<float>> obuffer_transpose(2); // complex
    size_t osize_transpose = compute_ptrdiff(olength_transpose, ostride_transpose);
    for(size_t i = 0; i < obuffer_transpose.size(); i++)
    {
        obuffer_transpose[i].resize(osize_transpose);
    }

    fill_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);
    // print_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);
    symmetrize_2d(obuffer_transpose, ilength, ostride_transpose);
    // print_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);

    // transpose back
    transpose(olength_transpose, ostride_transpose, axes);
    // print_buffer_2d(obuffer_transpose, olength_transpose, ostride_transpose);

    // impose expected format
    std::vector<std::vector<float>> obuffer(2);
    size_t                          osize = compute_ptrdiff(olength, ostride);
    for(size_t i = 0; i < obuffer.size(); i++)
    {
        obuffer[i].resize(osize);
    }

    size_t obuffer_bytes           = osize * sizeof(float);
    size_t obuffer_transpose_bytes = osize_transpose * sizeof(float);

    float* d_obuffer_transpose_real;
    assert(hipMalloc(&d_obuffer_transpose_real, obuffer_transpose_bytes) == hipSuccess);
    assert(hipMemcpy(d_obuffer_transpose_real,
                     obuffer_transpose[0].data(),
                     obuffer_transpose_bytes,
                     hipMemcpyHostToDevice)
           == hipSuccess);

    float* d_obuffer_transpose_imag;
    assert(hipMalloc(&d_obuffer_transpose_imag, obuffer_transpose_bytes) == hipSuccess);
    assert(hipMemcpy(d_obuffer_transpose_imag,
                     obuffer_transpose[1].data(),
                     obuffer_transpose_bytes,
                     hipMemcpyHostToDevice)
           == hipSuccess);

    float* d_obuffer_real;
    assert(hipMalloc(&d_obuffer_real, obuffer_bytes) == hipSuccess);

    float* d_obuffer_imag;
    assert(hipMalloc(&d_obuffer_imag, obuffer_bytes) == hipSuccess);

    hipLaunchKernelGGL(impose_expected_format_2d,
                       dim3(32, 32),
                       dim3(ceildiv(olength[1], 32), ceildiv(olength[0], 32)),
                       0,
                       0,
                       d_obuffer_transpose_real,
                       d_obuffer_transpose_imag,
                       d_obuffer_real,
                       d_obuffer_imag,
                       ostride_transpose[0],
                       ostride_transpose[1],
                       ostride[0],
                       ostride[1],
                       length[0],
                       length[1]);

    assert(hipMemcpy(obuffer[0].data(), d_obuffer_real, obuffer_bytes, hipMemcpyDeviceToHost)
           == hipSuccess);
    assert(hipMemcpy(obuffer[1].data(), d_obuffer_imag, obuffer_bytes, hipMemcpyDeviceToHost)
           == hipSuccess);

    // output
    print_buffer_2d(obuffer, olength, ostride);

    // Release device memory
    assert(hipFree(d_obuffer_transpose_real) == hipSuccess);
    assert(hipFree(d_obuffer_transpose_imag) == hipSuccess);

    assert(hipFree(d_obuffer_real) == hipSuccess);
    assert(hipFree(d_obuffer_imag) == hipSuccess);

    return 0;
}