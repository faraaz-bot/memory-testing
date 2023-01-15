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
* rocfft_mp_transpose: Multi-Process 3-D array transpose using advanced MPI.
*/

#ifndef ROCFFT_MP_TRANSPOSE
#define ROCFFT_MP_TRANSPOSE

#include <algorithm>
#include <complex.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <random>
#include <stdlib.h>
#include <vector>

enum rocfft_3D_slab_split
{
    ROCFFT_SLABS_SPLIT_X,
    ROCFFT_SLABS_SPLIT_Y,
    ROCFFT_SLABS_SPLIT_Z
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

// Create sequence of subarrays
void plan_transpose(std::vector<size_t> N,
                    int                 in_axis,
                    int                 out_axis,
                    MPI_Datatype        my_type,
                    MPI_Datatype*       subarrays_input,
                    MPI_Datatype*       subarrays_output,
                    MPI_Comm            comm)
{

    int my_rank, nprocs;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &nprocs);

    // Find axes that define slabs:
    std::vector<int> axes = {0, 1, 2};
    std::vector<int> slabs;

    std::vector<int>::iterator it = axes.begin();
    while((it = std::find_if(it, axes.end(), [in_axis](int x) { return x != in_axis; }))
          != axes.end())
    {
        slabs.push_back(std::distance(axes.begin(), it));
        it++;
    }

    int dim      = N.size();
    int mid_axis = dim - in_axis - out_axis;

    // Find local size for each process
    std::vector<int> in_shape(dim);
    std::vector<int> out_shape(dim);

    // input shapes
    int n, s;
    split_array(N[in_axis], nprocs, my_rank, &n, &s);

    in_shape[0] = n;
    in_shape[1] = N[mid_axis];
    in_shape[2] = N[out_axis];

    // output shapes
    split_array(N[out_axis], nprocs, my_rank, &n, &s);

    out_shape[0] = N[in_axis];
    out_shape[1] = N[mid_axis];
    out_shape[2] = n;

    // std::cout << "in_shape: " << in_shape[0] << ", " << in_shape[1]  << ",  " << in_shape[2] << std::endl;
    // std::cout << "out_shape: " << out_shape[0] << ", " << out_shape[1]  << ",  " << out_shape[2] << std::endl;

    // Create subarray sequences for intermediate transposition
    std::vector<int> dims_subarray(dim);
    std::vector<int> coords_subarray(dim, 0);

    // Subarrays for input array configuration
    std::copy(in_shape.begin(), in_shape.end(), dims_subarray.begin());

    for(unsigned int i = 0; i < nprocs; ++i)
    {
        split_array(in_shape[in_axis], nprocs, i, &n, &s);

        dims_subarray[in_axis]   = n;
        coords_subarray[in_axis] = s;

        MPI_Type_create_subarray(dim,
                                 in_shape.data(),
                                 dims_subarray.data(),
                                 coords_subarray.data(),
                                 MPI_ORDER_C,
                                 my_type,
                                 &subarrays_input[i]);

        MPI_Type_commit(&subarrays_input[i]);
    }

    // Subarrays for output array configuration
    std::copy(out_shape.begin(), out_shape.end(), dims_subarray.begin());
    std::fill(coords_subarray.begin(), coords_subarray.end(), 0);

    for(unsigned int i = 0; i < nprocs; ++i)
    {
        split_array(out_shape[out_axis], nprocs, i, &n, &s);

        dims_subarray[out_axis]   = n;
        coords_subarray[out_axis] = s;

        MPI_Type_create_subarray(dim,
                                 out_shape.data(),
                                 dims_subarray.data(),
                                 coords_subarray.data(),
                                 MPI_ORDER_C,
                                 my_type,
                                 &subarrays_output[i]);

        MPI_Type_commit(&subarrays_output[i]);
    }
}

#endif // ROCFFT_MP_TRANSPOSE
