//
// build:
//    hipcc st_lds_sim.cpp -o st_lds_sim -lboost_program_options
//
// quick test:
//    ./st_lds_sim -t 2 -f 4 2 -w 4
//

#include <boost/program_options.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>
namespace po = boost::program_options;

template <typename Titer>
typename Titer::value_type product(Titer begin, Titer end)
{
    return std::accumulate(
        begin, end, typename Titer::value_type(1), std::multiplies<typename Titer::value_type>());
}

void load_lds(
    int length, int threads_per_transform, int offset_lds, int threadIdx, int h, int width, int dt)
{
    int thread  = threadIdx % threads_per_transform;
    int lstride = 1;
    for(int w = 0; w < width; ++w)
    {
        const auto tid = thread + dt + h * threads_per_transform;
        const auto idx = offset_lds + (tid + w * length / width) * lstride;
        std::cout << "\tread  lds: R[" << h * width + w << "], lds[" << idx << "]" << std::endl;
    }
}

void store_lds(int length,
               int threads_per_transform,
               int offset_lds,
               int threadIdx,
               int h,
               int width,
               int dt,
               int cumheight)
{
    int thread  = threadIdx % threads_per_transform;
    int lstride = 1;

    for(int w = 0; w < width; ++w)
    {
        const auto tid = thread + dt + h * threads_per_transform;
        const auto idx
            = offset_lds
              + (tid / cumheight * (width * cumheight) + tid % cumheight + w * cumheight) * lstride;

        std::cout << "\twrite lds: lds[" << idx << "], R [" << h * width + w << "]" << std::endl;
    }
}

template <typename T>
void st_batched_1d_lds_conflict_sim(int               threads_per_transform,
                                    std::vector<int>& factors,
                                    int               wavefront_size,
                                    int               bank_bytes)
{
    int   length             = product(factors.begin(), factors.end());
    int   elem_bytes         = sizeof(T);
    float transform_per_warp = (float)wavefront_size / threads_per_transform;
    std::cout << "transform_per_warp: " << transform_per_warp << std::endl;

    // simulate the first n complete transforms only for now
    for(int transform_id = 0; transform_id < transform_per_warp; transform_id++)
    {
        std::cout << "Transform " << transform_id << std::endl;
        // no any padding or strides, elementwise
        int regular_offset_lds = transform_id * length;

        for(int npass = 0; npass < factors.size(); ++npass)
        {
            int   width     = factors[npass];
            float height    = static_cast<float>(length) / width / threads_per_transform;
            int   cumheight = product(factors.begin(), factors.begin() + npass);

            int iheight = std::floor(height);
            if(height > iheight && threads_per_transform > length / width)
                iheight += 1;
            std::cout << "  pass " << npass << std::endl;
            // std::cout << "iheight " << iheight << std::endl;
            if(npass != factors.size() - 1)
                for(int h = 0; h < iheight; ++h)
                    //work += generator(h, 0, width, 0);
                    for(int threadIdx = 0; threadIdx < threads_per_transform; ++threadIdx)
                        store_lds(length,
                                  threads_per_transform,
                                  regular_offset_lds,
                                  threadIdx,
                                  h,
                                  width,
                                  0,
                                  cumheight);

            if(npass != 0)
                for(int h = 0; h < iheight; ++h)
                    //work += generator(h, 0, width, 0);
                    for(int threadIdx = 0; threadIdx < threads_per_transform; ++threadIdx)
                        load_lds(length,
                                 threads_per_transform,
                                 regular_offset_lds,
                                 threadIdx,
                                 h,
                                 width,
                                 0);
        }
    }

    // if(threads_per_transform > wavefront_size)
    // {
    // }
    // else if(threads_per_transform == wavefront_size)
    // {
    //     // guarantee one wavefront doing 1 transform
    // }
    // else
    // {
    // }
}

int main(int argc, char* argv[])
{
    std::vector<int> factors;

    // Some exampls
    // factors.push_back(5);
    // factors.push_back(5);
    // factors.push_back(4);
    // st_batched_1d_lds_conflict_sim(10, factors, 64, 4);

    // factors.push_back(8);
    // factors.push_back(8);
    // st_batched_1d_lds_conflict_sim<float2>(8, factors, 64, 4);

    // factors.push_back(4);
    // factors.push_back(2);
    // st_batched_1d_lds_conflict_sim<float2>(2, factors, 4, 4);

    int         wavefront_size;
    int         bank_bytes;
    int         threads_per_transform;
    std::string precision;
    // clang-format off
    po::options_description opdesc("lds conflict sim options");
    opdesc.add_options()
        ("help,h", "produces this help message")
        ("precision,p",  po::value<std::string>(&precision)->default_value("float2"), "precision choices: float2, double2.")
        ("threads_per_transform,t", po::value<int>(&threads_per_transform)->default_value(4), "threads_per_transform")
        ("factors,f", po::value<std::vector<int>>(&factors)->multitoken(), "Radices to factorize the FFT.")
        ("wavefront_size,w", po::value<int>(&wavefront_size)->default_value(64), "wavefront_size")
        ("bank_bytes,b", po::value<int>(&bank_bytes)->default_value(4), "lds bank size in bytes");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return 0;
    }

    if(precision == "float2")
        st_batched_1d_lds_conflict_sim<float2>(
            threads_per_transform, factors, wavefront_size, bank_bytes);
    else if(precision == "double2")
        st_batched_1d_lds_conflict_sim<double2>(
            threads_per_transform, factors, wavefront_size, bank_bytes);

    return 0;
}