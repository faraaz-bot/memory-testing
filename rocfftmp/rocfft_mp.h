// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
* rocfft_mp: Multi-Process FFT computation with rocFFT backend and advanced MPI.
*/

#ifndef ROCFFT_MP
#define ROCFFT_MP

#include <algorithm>
#include <complex.h>
#include <fftw3-mpi.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <random>
#include <stdlib.h>
#include <vector>

#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"

enum rocfft_3D_slab_split
{
    ROCFFT_SLABS_SPLIT_X,
    ROCFFT_SLABS_SPLIT_Y,
    ROCFFT_SLABS_SPLIT_Z
};

struct rocfft_mp_plan_slabs
{
    // Default constructor
    rocfft_mp_plan_slabs(int np)
        : subarrays_input(new MPI_Datatype[np])
        , subarrays_output(new MPI_Datatype[np])
    {
    }

    std::vector<int> transpose_input_shape;
    std::vector<int> transpose_output_shape;

    MPI_Comm comm;
    int      nprocs;

    int axis;
    int dim_fast, dim_mid;

    int          nkia;
    MPI_Datatype data_type;

    MPI_Datatype* subarrays_input;
    MPI_Datatype* subarrays_output;

private:
    size_t local_fft_size;
};

void split_array(int N, int M, int p, int* n, int* s)
{
    int q = N / M;
    int r = N % M;
    *n    = q + (r > p);
    *s    = q * p + std::min(r, p);
}

// Geometry splitting for slab decomposition
std::vector<size_t> geometry_splitting(std::vector<size_t> N, int axis, MPI_Comm comm)
{
    int my_rank, nprocs;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &nprocs);

    size_t dim = N.size();

    // local_dims contains an extra entry to save the axis
    std::vector<size_t> local_dims(dim);

    std::copy(N.begin(), N.end(), local_dims.begin());

    // splitting array across the given axis
    int n, s;
    split_array(N[axis], nprocs, my_rank, &n, &s);
    local_dims[axis] = n;

    return local_dims;
}

// Print 3-D array for testing
void rocfft_mp_print_array(char*                              message,
                           int                                my_rank,
                           std::vector<std::complex<double>>& data,
                           int                                ld)
{
    printf("Proc[%d]: %s \n", my_rank, message);
    for(int i = 0; i < data.size(); i++)
    {
        std::cout << std::setw(20);
        std::cout << data[i] << " ";
        if((i + 1) % ld == 0)
            std::cout << "\n";
    }

    std::cout << std::endl;
    std::cout << std::endl;
}

// Parallel rocFFT plan
std::vector<rocfft_plan> rocfft_mp_plan_3D(const std::vector<size_t>& N,
                                           const std::vector<size_t>& local_dims,
                                           const int                  axis,
                                           const MPI_Datatype         my_type,
                                           std::vector<int>&          in_shape,
                                           std::vector<int>&          out_shape,
                                           MPI_Comm                   comm,
                                           rocfft_mp_plan_slabs&      options)
{

    int my_rank, nprocs;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &nprocs);

    // ---------------------------------------------
    // Plan_1: for slabs in the first two dimensions
    // ---------------------------------------------

    std::vector<rocfft_plan> plan_xyz;

    // Find axes that define slabs:
    std::vector<int> axes = {0, 1, 2};
    std::vector<int> slabs;

    std::vector<int>::iterator it = axes.begin();
    while((it = std::find_if(it, axes.end(), [axis](int x) { return x != axis; })) != axes.end())
    {
        slabs.push_back(std::distance(axes.begin(), it));
        it++;
    }

    // Length of transform:
    std::vector<size_t> length = {local_dims[slabs[0]], local_dims[slabs[1]]};

    // Set up the strides and buffer size for the input:
    std::vector<size_t> istride = {1};
    for(unsigned int i = 1; i < length.size(); ++i)
    {
        istride.push_back(length[i - 1] * istride[i - 1]);
    }
    const size_t isize = length[length.size() - 1] * istride[istride.size() - 1];

    // Set up the strides and buffer size for the output:
    std::vector<size_t> ostride = {1};
    for(unsigned int i = 1; i < length.size(); ++i)
    {
        ostride.push_back(length[i - 1] * ostride[i - 1]);
    }
    const size_t osize = length[length.size() - 1] * ostride[ostride.size() - 1];

    // Create plan description for computing a batch of slabs
    rocfft_plan_description desc_1 = NULL;
    rocfft_status           st     = rocfft_plan_description_create(&desc_1);
    if(st != rocfft_status_success)
        throw std::runtime_error("failed to create plan description for the first direction");

    st = rocfft_plan_description_set_data_layout(desc_1,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_complex_interleaved,
                                                 NULL,
                                                 NULL,
                                                 istride.size(), // input stride length
                                                 istride.data(), // input stride data
                                                 0, // input batch distance
                                                 ostride.size(), // output stride length
                                                 ostride.data(), // output stride data
                                                 0); // ouptut batch distance

    if(st != rocfft_status_success)
        throw std::runtime_error("failed to set data layout");

    // Create rocFFT plan for slabs in the first two dimensions
    rocfft_plan plan_1 = NULL;
    st                 = rocfft_plan_create(&plan_1,
                            rocfft_placement_notinplace,
                            rocfft_transform_type_complex_forward,
                            rocfft_precision_double,
                            length.size(), // Dimension
                            length.data(), // lengths
                            local_dims[axis], // Number of transforms
                            desc_1); // Description
    if(st != rocfft_status_success)
        throw std::runtime_error("failed to create plan for slabs");

    plan_xyz.push_back(plan_1);

    // -----------------------------------------
    // Plan_2: for pencils in the last dimension
    // -----------------------------------------

    // Length of transform:
    std::vector<size_t> length_pencils = {N[axis]};

    // Select axis for transposition, criterion: maximize overlap
    // We transpose from <axis> to <axis_fast>
    int axis_fast, axis_mid;
    if(N[slabs[0]] >= N[slabs[1]])
    {
        axis_fast = slabs[0];
        axis_mid  = slabs[1];
    }
    else
    {
        axis_fast = slabs[1];
        axis_mid  = slabs[0];
    }

    int n, s;
    split_array(N[axis], nprocs, my_rank, &n, &s);

    in_shape.push_back(n);
    in_shape.push_back(N[axis_mid]);
    in_shape.push_back(N[axis_fast]);

    // Calculate number of pencil transforms:
    split_array(N[axis_fast], nprocs, my_rank, &n, &s);

    out_shape.push_back(N[axis]);
    out_shape.push_back(N[axis_mid]);
    out_shape.push_back(n);

    size_t n_pencils_slow_dim = N[axis_mid] * n;

    // Set up the strides and buffer size for the input:
    std::vector<size_t> istride_pencils = {n_pencils_slow_dim};
    for(unsigned int i = 1; i < length_pencils.size(); ++i)
    {
        istride_pencils.push_back(length_pencils[i - 1] * istride_pencils[i - 1]);
    }

    // Set up the strides and buffer size for the output:
    std::vector<size_t> ostride_pencils = {n_pencils_slow_dim};
    for(unsigned int i = 1; i < length_pencils.size(); ++i)
    {
        ostride_pencils.push_back(length_pencils[i - 1] * ostride_pencils[i - 1]);
    }

    // Create plan description for computing a batch of pencils
    rocfft_plan_description desc_2 = NULL;
    st                             = rocfft_plan_description_create(&desc_2);
    if(st != rocfft_status_success)
        throw std::runtime_error("failed to create plan description for pencils (last dimension)");

    st = rocfft_plan_description_set_data_layout(desc_2,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_complex_interleaved,
                                                 NULL,
                                                 NULL,
                                                 istride_pencils.size(), // input stride length
                                                 istride_pencils.data(), // input stride data
                                                 1, // input batch distance
                                                 ostride_pencils.size(), // output stride length
                                                 ostride_pencils.data(), // output stride data
                                                 1); // ouptut batch distance
    if(st != rocfft_status_success)
        throw std::runtime_error("failed to set data layout");

    // Create rocFFT plan for slabs in the first two dimensions
    rocfft_plan plan_2 = NULL;
    st                 = rocfft_plan_create(&plan_2,
                            rocfft_placement_notinplace,
                            rocfft_transform_type_complex_forward,
                            rocfft_precision_double,
                            length_pencils.size(), // Dimension
                            length_pencils.data(), // lengths
                            n_pencils_slow_dim, // Number of transforms
                            desc_2); // Description
    if(st != rocfft_status_success)
        throw std::runtime_error("failed to create plan for the last dimension");

    plan_xyz.push_back(plan_2);

    rocfft_plan_description_destroy(desc_1);
    rocfft_plan_description_destroy(desc_2);

    options.transpose_input_shape  = in_shape;
    options.transpose_output_shape = out_shape;
    options.comm                   = comm;
    options.axis                   = axis;
    options.dim_fast               = axis_fast;
    options.dim_mid                = axis_mid;
    options.data_type              = my_type;
    options.nprocs                 = nprocs;

    // Create subarray sequences for intermediate transposition
    int              dim = in_shape.size();
    std::vector<int> dims_subarray(dim);
    std::vector<int> coords_subarray(dim, 0);

    // Subarrays for input array configuration
    std::copy(in_shape.begin(), in_shape.end(), dims_subarray.begin());

    for(unsigned int i = 0; i < nprocs; ++i)
    {
        split_array(in_shape[axis], nprocs, i, &n, &s);

        dims_subarray[axis]   = n;
        coords_subarray[axis] = s;

        MPI_Type_create_subarray(dim,
                                 in_shape.data(),
                                 dims_subarray.data(),
                                 coords_subarray.data(),
                                 MPI_ORDER_C,
                                 my_type,
                                 &options.subarrays_input[i]);

        MPI_Type_commit(&options.subarrays_input[i]);
    }

    // Subarrays for output array configuration
    std::copy(out_shape.begin(), out_shape.end(), dims_subarray.begin());
    std::fill(coords_subarray.begin(), coords_subarray.end(), 0);

    for(unsigned int i = 0; i < nprocs; ++i)
    {
        split_array(out_shape[0], nprocs, i, &n, &s);

        dims_subarray[0]   = n;
        coords_subarray[0] = s;

        MPI_Type_create_subarray(dim,
                                 out_shape.data(),
                                 dims_subarray.data(),
                                 coords_subarray.data(),
                                 MPI_ORDER_C,
                                 my_type,
                                 &options.subarrays_output[i]);

        MPI_Type_commit(&options.subarrays_output[i]);
    }

    return plan_xyz;
}

void subarray_sequence(MPI_Datatype datatype,
                       int          dimension,
                       int          dims_array[3],
                       int          axis,
                       int          nprocs,
                       MPI_Datatype subarrays[2])
{
    int dims_subarray[3], coords_subarray[3], n, s;

    for(int i = 0; i < dimension; i++)
    {
        dims_subarray[i]   = dims_array[i];
        coords_subarray[i] = 0;
    }

    for(int p = 0; p < nprocs; p++)
    {
        split_array(dims_array[axis], nprocs, p, &n, &s);

        dims_subarray[axis]   = n;
        coords_subarray[axis] = s;

        MPI_Type_create_subarray(dimension,
                                 dims_array,
                                 dims_subarray,
                                 coords_subarray,
                                 MPI_ORDER_C,
                                 datatype,
                                 &subarrays[p]);

        MPI_Type_commit(&subarrays[p]);
    }
}

// Parallel rocFFT execution
void rocfft_mp_execute(const std::vector<rocfft_plan>&    plan_xyz,
                       std::vector<std::complex<double>>& input,
                       std::vector<std::complex<double>>& output,
                       rocfft_mp_plan_slabs&              options)
{

    // Create HIP device buffer and copy data to device
    size_t local_fftsize = input.size();
    size_t Nbytes        = local_fftsize * sizeof(double2);

    float2 *data_device, *output_device;
    hipMalloc(&data_device, Nbytes);
    hipMalloc(&output_device, Nbytes);

    hipMemcpy(data_device, input.data(), Nbytes, hipMemcpyHostToDevice);
    hipMemcpy(output_device, output.data(), Nbytes, hipMemcpyHostToDevice);

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    rocfft_plan_get_work_buffer_size(plan_xyz[0], &work_buf_size);
    void*                 work_buf = nullptr;
    rocfft_execution_info info     = nullptr;

    if(work_buf_size)
    {
        rocfft_execution_info_create(&info);
        hipMalloc(&work_buf, work_buf_size);
        rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
    }

    // Execute plan
    rocfft_execute(plan_xyz[0], (void**)&data_device, (void**)&output_device, info);

    // Wait for execution to finish
    hipDeviceSynchronize();

    // Clean up work buffer
    if(work_buf_size)
    {
        hipFree(work_buf);
        rocfft_execution_info_destroy(info);
    }

    // Copying result back to host
    hipMemcpy(output.data(), output_device, Nbytes, hipMemcpyDeviceToHost);

    // 3-D transposition
    std::vector<int> counts(options.nprocs, 1);
    std::vector<int> displacements(options.nprocs, 0);

    MPI_Alltoallw(output.data(),
                  counts.data(),
                  displacements.data(),
                  options.subarrays_input,
                  input.data(),
                  counts.data(),
                  displacements.data(),
                  options.subarrays_output,
                  options.comm);

    // Moving transposed array to the GPU
    hipMemcpy(data_device, input.data(), Nbytes, hipMemcpyHostToDevice);

    // Check if the plan requires a work buffer
    size_t work_buf_size_2 = 0;
    rocfft_plan_get_work_buffer_size(plan_xyz[1], &work_buf_size_2);
    void*                 work_buf_2 = nullptr;
    rocfft_execution_info info_2     = nullptr;

    if(work_buf_size_2)
    {
        rocfft_execution_info_create(&info_2);
        hipMalloc(&work_buf_2, work_buf_size_2);
        rocfft_execution_info_set_work_buffer(info_2, work_buf_2, work_buf_size_2);
    }

    // Execute plan
    rocfft_execute(plan_xyz[1], (void**)&data_device, (void**)&output_device, info_2);

    // Wait for execution to finish
    hipDeviceSynchronize();

    // Clean up work buffer
    if(work_buf_size_2)
    {
        hipFree(work_buf_2);
        rocfft_execution_info_destroy(info_2);
    }

    // Copying result back to host
    hipMemcpy(output.data(), output_device, Nbytes, hipMemcpyDeviceToHost);

    // Transpose back to original shape: final_alignment -> initial_alignment
    MPI_Alltoallw(output.data(),
                  counts.data(),
                  displacements.data(),
                  options.subarrays_output,
                  input.data(),
                  counts.data(),
                  displacements.data(),
                  options.subarrays_input,
                  options.comm);

    hipFree(data_device);
    hipFree(output_device);
}

void rocfft_mpi_plan_destroy(const std::vector<rocfft_plan>& plan_xyz)
{
    for(unsigned int i = 0; i < plan_xyz.size(); ++i)
    {
        rocfft_plan_destroy(plan_xyz[i]);
    }
}

#endif // ROCFFT_MP
