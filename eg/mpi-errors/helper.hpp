#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        hipError_t error = (cmd);                                                              \
        if(error != hipSuccess)                                                                \
        {                                                                                      \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << "\n";                          \
            exit(-1);                                                                          \
        }                                                                                      \
    } while(0)

// RAII struct for single device buffer
class gpubuf
{
private:
    size_t N;
    float* buf;

public:
    gpubuf(size_t N)
    {
        HIP_CHECK(hipMalloc(&buf, sizeof(float) * N));
        HIP_CHECK(hipMemset(buf, 0, sizeof(float) * N));
    }

    ~gpubuf()
    {
        HIP_CHECK(hipFree(buf));
    }

    float* data()
    {
        return buf;
    }

    const size_t size()
    {
        return N;
    }
};
