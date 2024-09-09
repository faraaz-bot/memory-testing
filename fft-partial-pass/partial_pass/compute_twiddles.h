#ifndef COMPUTE_TWIDDLES_H
#define COMPUTE_TWIDDLES_H

#include "launch_common.h"

#include "rocfft/twiddle_factors.h"

void GetKernelParams(const std::vector<size_t> &radices,
                     std::vector<size_t> &radices_prod,
                     std::vector<size_t> &radices_sum_prod,
                     size_t &max_radix_prod,
                     size_t &min_radix,
                     size_t &table_sz)
{
    radices_sum_prod = {0};
    radices_prod = {};

    size_t prod = 1;
    size_t prod_next = radices.at(0);
    size_t sum = 0;

    for (size_t i = 0; i < radices.size() - 1; ++i)
    {
        auto radix = radices.at(i);
        auto radix_next = radices.at(i + 1);

        prod *= radix;
        prod_next *= radix_next;
        sum += prod * (radix_next - 1);
        radices_sum_prod.push_back(sum);
        radices_prod.push_back(prod_next);
    }

    if (radices_prod.empty())
        radices_prod.push_back(radices.at(0));

    max_radix_prod = *(std::max_element(std::begin(radices_prod), std::end(radices_prod)));
    min_radix = *(std::min_element(std::begin(radices), std::end(radices)));

    auto M = radices.size() - 1;
    auto last_radix = radices.at(M);
    table_sz = M ? radices_sum_prod.at(M - 1) + ((radices_prod.at(M - 1) / last_radix) - 1) * (last_radix - 1) + (last_radix - 1)
                 : radices_sum_prod.at(0);
}

template <typename T>
void ComputeTwiddles(size_t N, std::vector<size_t> &radices, gpubuf_t<T> &twiddles)
{
    size_t table_sz, maxElem, minElem;
    std::vector<size_t> radices_prod, radices_sum_prod;

    GetKernelParams(radices, radices_prod, radices_sum_prod, maxElem, minElem, table_sz);

    table_sz = std::min(table_sz, N);

    auto table_bytes = table_sz * sizeof(T);

    if (table_bytes == 0)
        return;

    auto ret = twiddles.alloc(table_bytes);

    auto num_radices = radices.size();

    auto blockSize = TWIDDLES_THREADS;
    auto numBlocksX = DivRoundingUp<size_t>(num_radices, blockSize);
    auto numBlocksY = DivRoundingUp<size_t>(maxElem / minElem, blockSize);

    auto device_data_ptr = static_cast<T *>(twiddles.data());

    radices_t radices_device;
    radices_t radices_prod_device;
    radices_t radices_sum_prod_device;
    std::copy(radices.begin(), radices.end(), radices_device.data);
    std::copy(radices_prod.begin(), radices_prod.end(), radices_prod_device.data);
    std::copy(radices_sum_prod.begin(), radices_sum_prod.end(), radices_sum_prod_device.data);

    hipLaunchKernelGGL(GenerateTwiddleTableKernel<T>,
                       dim3(numBlocksX, numBlocksY),
                       dim3(blockSize, blockSize),
                       0, // sharedMemBytes
                       0,
                       N,
                       num_radices,
                       radices_device,
                       radices_prod_device,
                       radices_sum_prod_device,
                       device_data_ptr);

    ret = hipDeviceSynchronize();
}

template <typename T>
void ComputeTwiddlesLarge(size_t N, size_t largeTwdBase, gpubuf_t<T> &twiddles_large)
{
    auto X = static_cast<size_t>(1) << largeTwdBase; // ex: 2^8 = 256
    auto Y = DivRoundingUp<size_t>(CeilPo2(N), largeTwdBase);
    auto tableSize = X * Y;

    auto table_bytes = tableSize * sizeof(T);
    if (twiddles_large.alloc(table_bytes) != hipSuccess)
        throw std::runtime_error("unable to allocate twiddle length " + std::to_string(tableSize));

    if (table_bytes == 0)
        return;

    auto blockSize = TWIDDLES_THREADS;

    double phi = TWO_PI / double(N);

    auto numBlocksX = DivRoundingUp<size_t>(X, blockSize);
    auto numBlocksY = DivRoundingUp<size_t>(Y, blockSize);

    hipLaunchKernelGGL(GenerateTwiddleTableLargeKernel<T>,
                       dim3(numBlocksX, numBlocksY),
                       dim3(blockSize, blockSize),
                       0, // sharedMemBytes
                       0,
                       phi,
                       largeTwdBase,
                       X,
                       Y,
                       static_cast<T *>(twiddles_large.data()));

    auto ret = hipDeviceSynchronize();
}

template <typename T>
void ComputeTwiddles2D(size_t N1,
                       size_t N2,
                       std::vector<size_t> &radices1,
                       std::vector<size_t> &radices2,
                       gpubuf_t<T> &twiddles)
{
    if (radices1 == radices2)
        N2 = 0;

    auto blockSize = TWIDDLES_THREADS;

    size_t table_sz_1, maxElem_1, minElem_1;
    std::vector<size_t> radices_prod_1, radices_sum_prod_1;

    GetKernelParams(
        radices1, radices_prod_1, radices_sum_prod_1, maxElem_1, minElem_1, table_sz_1);

    size_t table_sz_2, maxElem_2, minElem_2;
    std::vector<size_t> radices_prod_2, radices_sum_prod_2;

    if (N2)
        GetKernelParams(
            radices2, radices_prod_2, radices_sum_prod_2, maxElem_2, minElem_2, table_sz_2);
    else
        table_sz_2 = N2;

    auto table_sz = (table_sz_1 + table_sz_2);
    auto table_bytes = table_sz * sizeof(T);

    if (table_bytes == 0)
        return;

    if (twiddles.alloc(table_bytes) != hipSuccess)
        throw std::runtime_error("unable to allocate twiddle length " + std::to_string(table_sz));

    auto device_data_ptr = static_cast<T *>(twiddles.data());

    auto num_radices_1 = radices1.size();

    auto numBlocksX_1 = DivRoundingUp<size_t>(num_radices_1, blockSize);
    auto numBlocksY_1 = DivRoundingUp<size_t>(maxElem_1 / minElem_1, blockSize);

    radices_t radices1_device;
    radices_t radices_prod_device_1;
    radices_t radices_sum_prod_device_1;
    std::copy(radices1.begin(), radices1.end(), radices1_device.data);
    std::copy(radices_prod_1.begin(), radices_prod_1.end(), radices_prod_device_1.data);
    std::copy(
        radices_sum_prod_1.begin(), radices_sum_prod_1.end(), radices_sum_prod_device_1.data);

    hipLaunchKernelGGL(GenerateTwiddleTableKernel<T>,
                       dim3(numBlocksX_1, numBlocksY_1),
                       dim3(blockSize, blockSize),
                       0, // sharedMemBytes
                       0,
                       N1,
                       num_radices_1,
                       radices1_device,
                       radices_prod_device_1,
                       radices_sum_prod_device_1,
                       device_data_ptr);

    if (N2)
    {
        auto num_radices_2 = radices2.size();

        auto numBlocksX_2 = DivRoundingUp<size_t>(num_radices_2, blockSize);
        auto numBlocksY_2 = DivRoundingUp<size_t>(maxElem_2 / minElem_2, blockSize);

        radices_t radices2_device;
        radices_t radices_prod_device_2;
        radices_t radices_sum_prod_device_2;
        std::copy(radices2.begin(), radices2.end(), radices2_device.data);
        std::copy(radices_prod_2.begin(), radices_prod_2.end(), radices_prod_device_2.data);
        std::copy(radices_sum_prod_2.begin(),
                  radices_sum_prod_2.end(),
                  radices_sum_prod_device_2.data);

        hipLaunchKernelGGL(GenerateTwiddleTableKernel<T>,
                           dim3(numBlocksX_2, numBlocksY_2),
                           dim3(blockSize, blockSize),
                           0, // sharedMemBytes
                           0,
                           N2,
                           num_radices_2,
                           radices2_device,
                           radices_prod_device_2,
                           radices_sum_prod_device_2,
                           device_data_ptr + table_sz_1);
    }
}

#endif