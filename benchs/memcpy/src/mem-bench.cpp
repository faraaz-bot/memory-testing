#include "mem-bench.hpp"

void run_memcpy(const int N, const std::vector<float*>& in_bufs, std::vector<float*>& out_bufs)
{
    const size_t ngpus = in_bufs.size();
    const size_t sub_block_size = N / ngpus; // Length of block in each transfer
    const size_t bytes_to_copy_per_row = sub_block_size * sizeof(float); // Bytes per row in transfer
    const size_t pitch_bytes = N * sizeof(float); // Width of buf

    float ms;
    hipEvent_t start, end;
    for(auto i = 0; i < ngpus; i++) // src GPU
    {
        for(auto j = 0; j < ngpus; j++) // Offset within GPU, AKA dst GPU
        {
            HIP_CHECK(hipMemcpy2D(out_bufs[j] + (i * sub_block_size), pitch_bytes, in_bufs[i] + (j * sub_block_size), pitch_bytes, bytes_to_copy_per_row, sub_block_size, hipMemcpyDeviceToDevice));
        }
    }
    return;
}


// (1.2) hipMemcpy2D between two devices, using streams
void run_memcpy_async(const int N, const std::vector<float*>& in_bufs, std::vector<float*>& out_bufs, const std::vector<hipStream_t>& streams)
{
    return;
}


// (4) RCCL alltoall

/* Helpers for verifying correctness */

// Combine ngpu # of gpubuf partitions back in an N x N matrix on the host
// Assumes hostbuf_result has enough memory allocated for it
void assemble_output_to_host(const int N, const std::vector<float*>& gpubufs, float* hostbuf_result)
{
    const size_t ngpus = gpubufs.size();
    const size_t buf_size = N * N / ngpus;
    for(auto i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipMemcpy(hostbuf_result + i * buf_size, gpubufs[i], buf_size * sizeof(float), hipMemcpyDeviceToHost));
    }

}

// Helper just to print N consecutive values in gpubuf
__global__ void print(const int N, const float* input)
{
    printf("[ ");
    for(int i = 0; i < N; i++)
        printf("%.6f ", input[i]);
    printf("]\n");
}

// Helper just to print N consecutive values in gpubuf
__global__ void print2d(const int N, const int M, const float* input)
{
    printf("[\n");
    for(int i = 0; i < N; i++)
    {
        printf("\t[ ");
        for(int j = 0; j < M; j++)
        {
            auto idx = i + N * j;
            printf("%.6f ", input[idx]);
        }
        printf("]\n");
    }
    printf(" ]\n");
}

// Check equality of matrices
bool is_same_matrix(const int N, const std::vector<float>& input1, const std::vector<float>& input2)
{
    for (auto i = 0; i < N*N; i++)
        if(input1[i] != input2[i]) return false;

    return true;
}


// Helper to print initial host matrix and transposed matrix
void print_host_2d(const int N, const int M, const std::vector<float>& input)
{
    std::cout << "[\n";
    for(int i = 0; i < N; i++)
    {
        std::cout << "  [ ";
        for(int j = 0; j < M; j++)
        {
            auto idx = i * N + j;
            std::cout << std::setw(4) << input[idx] << " ";
        }
        std::cout << " ]\n";
    }
    std::cout << "]" << std::endl;
}

// Reference impl (out-of-place)
void host_transpose(const int N, const std::vector<float>& input, std::vector<float>& output)
{
    output.reserve(N*N);
#pragma omp parallel for
    for(size_t i = 0; i < N; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            // Get curr index and send data to opposing location across diagonal
            auto idx1 = i * N + j;
            auto idx2 = j * N + i;
            output[idx2] = input[idx1];
        }
    }
}