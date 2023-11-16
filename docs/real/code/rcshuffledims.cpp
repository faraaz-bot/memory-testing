// usage: hipcc rcshuffledims.cpp -lrocfft

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft/rocfft.h>

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
template <typename Tval>
__global__ void resymmetrize_2d(const Tval*  idata,
                                Tval*        odata,
                                const size_t istride_x,
                                const size_t istride_y,
                                const size_t ostride_x,
                                const size_t ostride_y,
                                const size_t length_x,
                                const size_t length_y)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < length_x / 2 + 1)
    {
        if(iy < length_y / 2 + 1)
        {
            const size_t gopos = ix * ostride_x + iy * ostride_y;
            const size_t gipos = ix * istride_x + iy * istride_y;

            odata[gopos] = idata[gipos];
        }
        else if(iy < length_y)
        {
            const size_t gopos = ix * ostride_x + iy * ostride_y;
            const size_t gipos
                = (ix == 0 ? 0 : length_x - ix) * istride_x + (length_y - iy) * istride_y;

            odata[gopos].x = idata[gipos].x;
            odata[gopos].y = -idata[gipos].y;
        }
    }
}
// TO DO: resymmetrized_transpose_3d

// overload for real
void fill_buffer(std::vector<float>&        buffer,
                 const std::vector<size_t>& length,
                 const std::vector<size_t>& stride)
{
    for(size_t i = 0; i < length[0]; ++i)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            if(length.size() == 2)
            {
                size_t idx  = i * stride[0] + j * stride[1];
                buffer[idx] = sin(i * 1 - j * 2) + cos((i + 3) * (j + 4));
            }
            else if(length.size() == 3)
            {
                for(size_t k = 0; k < length[2]; ++k)
                {
                    size_t idx = i * stride[0] + j * stride[1] + k * stride[2];
                    buffer[idx]
                        = sin((i * 1 - j * 2) + (k / 5)) + cos(((i + 3) * (j + 4)) - (k * 6));
                }
            }
        }
    }
}
// overload for complex
void fill_buffer(std::vector<float2>&       buffer,
                 const std::vector<size_t>& length,
                 const std::vector<size_t>& stride)
{
    for(size_t i = 0; i < length[0]; ++i)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            if(length.size() == 2)
            {
                size_t idx    = i * stride[0] + j * stride[1];
                buffer[idx].x = sin(i * 1 - j * 2) + cos((i + 3) * (j + 4));
                buffer[idx].y = sin(i * 4 - j * 3) + cos((i + 2) * (j + 1));
            }
            else if(length.size() == 3)
            {
                for(size_t k = 0; k < length[2]; ++k)
                {
                    size_t idx = i * stride[0] + j * stride[1] + k * stride[2];
                    buffer[idx].x
                        = sin((i * 1 - j * 2) + (k / 5)) + cos(((i + 3) * (j + 4)) - (k * 6));
                    buffer[idx].y
                        = sin((i * 4 - j * 3) + (k / 6)) + cos(((i + 2) * (j + 1)) - (k * 5));
                }
            }
        }
    }
}

// overload for real
void print_buffer(const std::vector<float>&  buffer,
                  const std::vector<size_t>& length,
                  const std::vector<size_t>& stride)
{
    if(length.size() == 2)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            for(size_t i = 0; i < length[0]; ++i)
            {
                size_t idx = i * stride[0] + j * stride[1];

                std::stringstream ss;
                ss << std::noshowpos << buffer[idx] << std::flush;

                std::cout << std::left << std::setw(12) << std::setfill(' ') << ss.str()
                          << std::flush;
            }
            std::cout << std::endl;
        }
    }
    else if(length.size() == 3)
    {
        for(size_t k = 0; k < length[2]; ++k)
        {
            for(size_t j = 0; j < length[1]; ++j)
            {
                for(size_t i = 0; i < length[0]; ++i)
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
    }
    std::cout << std::endl;
}
// overload for complex
void print_buffer(const std::vector<float2>& buffer,
                  const std::vector<size_t>& length,
                  const std::vector<size_t>& stride)
{
    if(length.size() == 2)
    {
        for(size_t j = 0; j < length[1]; ++j)
        {
            for(size_t i = 0; i < length[0]; ++i)
            {
                size_t idx = i * stride[0] + j * stride[1];

                std::stringstream ss;
                ss << std::noshowpos << buffer[idx].x << std::showpos << buffer[idx].y << "j"
                   << std::flush;

                std::cout << std::left << std::setw(21) << std::setfill(' ') << ss.str()
                          << std::flush;
            }
            std::cout << std::endl;
        }
    }
    else if(length.size() == 3)
    {
        for(size_t k = 0; k < length[2]; ++k)
        {
            for(size_t j = 0; j < length[1]; ++j)
            {
                for(size_t i = 0; i < length[0]; ++i)
                {
                    size_t idx = i * stride[0] + j * stride[1];

                    std::stringstream ss;
                    ss << std::noshowpos << buffer[idx].x << std::showpos << buffer[idx].y << "j"
                       << std::flush;

                    std::cout << std::left << std::setw(21) << std::setfill(' ') << ss.str()
                              << std::flush;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

size_t compute_ptrdiff(const std::vector<size_t>& length, const std::vector<size_t>& stride)
{
    size_t val = 1;
    for(size_t i = 0; i < length.size(); ++i)
    {
        val += (length[i] - 1) * stride[i];
    }
    return val;
}

// rocFFT gpu compute
void rocfft_r2c(const std::vector<float>&  ibuffer,
                std::vector<float2>&       obuffer,
                const size_t               isize,
                const size_t               osize,
                const std::vector<size_t>& length,
                const std::vector<size_t>& olength,
                const std::vector<size_t>& istride,
                const std::vector<size_t>& ostride,
                size_t                     nbatch)
{
    rocfft_setup();

    size_t ibytes = isize * sizeof(float);
    size_t obytes = osize * sizeof(float2);

    // Create HIP device buffers and copy data to device
    float* d_ibuffer;
    hipMalloc(&d_ibuffer, ibytes);
    hipMemcpy(d_ibuffer, ibuffer.data(), ibytes, hipMemcpyHostToDevice);

    float2* d_obuffer;
    hipMalloc(&d_obuffer, obytes);

    // Create description struct
    rocfft_plan_description description = nullptr;
    rocfft_plan_description_create(&description);
    rocfft_plan_description_set_data_layout(description,
                                            rocfft_array_type_real,
                                            rocfft_array_type_hermitian_interleaved,
                                            nullptr,
                                            nullptr,
                                            istride.size(),
                                            istride.data(),
                                            0,
                                            ostride.size(),
                                            ostride.data(),
                                            0);

    // Create rocFFT plan
    rocfft_plan plan = nullptr;
    rocfft_plan_create(&plan,
                       rocfft_placement_notinplace,
                       rocfft_transform_type_real_forward,
                       rocfft_precision_single,
                       length.size(),
                       length.data(),
                       nbatch,
                       description);

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    rocfft_plan_get_work_buffer_size(plan, &work_buf_size);
    void*                 work_buf = nullptr;
    rocfft_execution_info info     = nullptr;
    if(work_buf_size)
    {
        rocfft_execution_info_create(&info);
        hipMalloc(&work_buf, work_buf_size);
        rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
    }

    // Execute plan
    rocfft_execute(plan, (void**)&d_ibuffer, (void**)&d_obuffer, info);

    // Wait for execution to finish
    hipDeviceSynchronize();

    // Clean up work buffer
    if(work_buf_size)
    {
        hipFree(work_buf);
        rocfft_execution_info_destroy(info);
    }

    // Copy result back to host
    hipMemcpy(obuffer.data(), d_obuffer, obytes, hipMemcpyDeviceToHost);

    // Free device buffer
    hipFree(d_ibuffer);
    hipFree(d_obuffer);

    // Destroy plans
    rocfft_plan_description_destroy(description);
    rocfft_plan_destroy(plan);
    plan = nullptr;

    rocfft_cleanup();
}
// TO DO: rocFFT C2C

void test_2d()
{
    std::vector<size_t> length  = {5, 6};
    std::vector<size_t> istride = {1, length[0]};

    std::vector<size_t> olength = {length[0] / 2 + 1, length[1]};
    std::vector<size_t> ostride = {1, olength[0]};

    size_t             isize = compute_ptrdiff(length, istride);
    std::vector<float> ibuffer(isize);

    // initialize data
    fill_buffer(ibuffer, length, istride);

    // no need to transpose
    if(length[1] % 2 != 0 && length[0] % 2 == 0
       || istride[1] > istride[0] && (length[1] % 2 != 0 || length[0] % 2 == 0))
    {
        std::cout << "No need to transpose" << std::endl;
        return;
    }

    // transpose directions
    std::vector<size_t> length_transpose  = {length[1], length[0]};
    std::vector<size_t> istride_transpose = {istride[1], istride[0]};

    // execute FFT
    std::vector<size_t> olength_transpose = {length_transpose[0] / 2 + 1, length_transpose[1]};
    std::vector<size_t> ostride_transpose = {1, olength_transpose[0]};

    size_t              osize_alternative = compute_ptrdiff(olength_transpose, ostride_transpose);
    std::vector<float2> obuffer_alternative(osize_alternative);

    rocfft_r2c(ibuffer,
               obuffer_alternative,
               isize,
               osize_alternative,
               length_transpose,
               olength_transpose,
               istride_transpose,
               ostride_transpose,
               1);

    // transpose back (currently in alternative data format)
    std::vector<size_t> olength_alternative = {olength_transpose[1], olength_transpose[0]};
    std::vector<size_t> ostride_alternative = {ostride_transpose[1], ostride_transpose[0]};

    // transpose back and impose expected format
    size_t              osize = compute_ptrdiff(olength, ostride);
    std::vector<float2> obuffer(osize);

    size_t obuffer_alternative_bytes = osize * sizeof(float2);
    size_t obuffer_bytes             = osize * sizeof(float2);

    void* d_obuffer_alternative;
    assert(hipMalloc(&d_obuffer_alternative, obuffer_alternative_bytes) == hipSuccess);
    assert(hipMemcpy(d_obuffer_alternative,
                     obuffer_alternative.data(),
                     obuffer_alternative_bytes,
                     hipMemcpyHostToDevice)
           == hipSuccess);

    void* d_obuffer;
    assert(hipMalloc(&d_obuffer, obuffer_bytes) == hipSuccess);

    hipLaunchKernelGGL(resymmetrize_2d,
                       dim3(32, 32),
                       dim3(ceildiv(olength[1], 32), ceildiv(olength[0], 32)),
                       0,
                       0,
                       (float2*)d_obuffer_alternative,
                       (float2*)d_obuffer,
                       ostride_alternative[0],
                       ostride_alternative[1],
                       ostride[0],
                       ostride[1],
                       length[0],
                       length[1]);

    assert(hipMemcpy(obuffer.data(), d_obuffer, obuffer_bytes, hipMemcpyDeviceToHost)
           == hipSuccess);

    assert(hipFree(d_obuffer_alternative) == hipSuccess);
    assert(hipFree(d_obuffer) == hipSuccess);

    // output
    print_buffer(obuffer, olength, ostride);
}
void test_3d()
{
    // NOT IMPLEMENTED
}

int main()
{
    test_2d();
    // test_3d();

    return 0;
}