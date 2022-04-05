//
// build:
//    hipcc st_lds_sim.cpp -o st_lds_sim -lboost_program_options
//
// quick tests:
//    ./st_lds_sim -t 2 -f 4 2 -w 4 -v
//    ./st_lds_sim -t 8 -f 8 8
//    ./st_lds_sim -t 10 -f 5 5 4
//

#include <boost/program_options.hpp>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
namespace po = boost::program_options;

template <typename Titer>
typename Titer::value_type product(Titer begin, Titer end)
{
    return std::accumulate(
        begin, end, typename Titer::value_type(1), std::multiplies<typename Titer::value_type>());
}

std::vector<std::tuple<int, int>> load_lds(int  length,
                                           int  threads_per_transform,
                                           int  offset_lds,
                                           int  threadIdx,
                                           int  h,
                                           int  width,
                                           int  dt,
                                           int  elem_bytes,
                                           int  bank_bytes,
                                           bool debug)
{
    std::vector<std::tuple<int, int>> ret;
    int                               thread  = threadIdx % threads_per_transform;
    int                               lstride = 1;
    for(int w = 0; w < width; ++w)
    {
        const auto tid = thread + dt + h * threads_per_transform;
        const auto idx = offset_lds + (tid + w * length / width) * lstride;

        const auto bank_start = idx * elem_bytes / bank_bytes;
        const auto bank_end   = ((idx + 1) * elem_bytes - 1) / bank_bytes;
        ret.push_back(std::make_tuple(bank_start, bank_end));
        if(debug)
            std::cout << "\ttid " << std::setw(3) << threadIdx << " R: reg[" << std::setw(3)
                      << h * width + w << "], lds[" << std::setw(3) << idx << "], bank["
                      << std::setw(2) << bank_start << "-" << std::setw(2) << bank_end << "]"
                      << std::endl;
    }
    return ret;
}

std::vector<std::tuple<int, int>> store_lds(int  length,
                                            int  threads_per_transform,
                                            int  offset_lds,
                                            int  threadIdx,
                                            int  h,
                                            int  width,
                                            int  dt,
                                            int  cumheight,
                                            int  elem_bytes,
                                            int  bank_bytes,
                                            bool debug)
{
    std::vector<std::tuple<int, int>> ret;
    int                               thread  = threadIdx % threads_per_transform;
    int                               lstride = 1;

    for(int w = 0; w < width; ++w)
    {
        const auto tid = thread + dt + h * threads_per_transform;
        const auto idx
            = offset_lds
              + (tid / cumheight * (width * cumheight) + tid % cumheight + w * cumheight) * lstride;

        const auto bank_start = idx * elem_bytes / bank_bytes;
        const auto bank_end   = ((idx + 1) * elem_bytes - 1) / bank_bytes;
        ret.push_back(std::make_tuple(bank_start, bank_end));
        if(debug)
            std::cout << "\ttid " << std::setw(3) << threadIdx << " W: lds[" << std::setw(3) << idx
                      << "], reg[" << std::setw(3) << h * width + w << "], bank[" << std::setw(2)
                      << bank_start << "-" << std::setw(2) << bank_end << "]" << std::endl;
    }
    return ret;
}

template <typename T>
void st_batched_1d_lds_conflict_sim(int               threads_per_transform,
                                    std::vector<int>& factors,
                                    int               wavefront_size,
                                    int               bank_bytes,
                                    bool              debug)
{
    int   length             = product(factors.begin(), factors.end());
    int   elem_bytes         = sizeof(T);
    float transform_per_warp = (float)wavefront_size / threads_per_transform;
    std::cout << "transform_per_warp: " << transform_per_warp << std::endl;

    for(int npass = 0; npass < factors.size(); ++npass)
    {
        std::cout << "Pass " << npass << std::endl;
        // simulate the first n complete transforms only for now
        for(int transform_id = 0; transform_id < transform_per_warp; transform_id++)
        {
            std::cout << "\ttransform " << transform_id << std::endl;
            // no any padding or strides, elementwise
            int regular_offset_lds = transform_id * length;

            int   width     = factors[npass];
            float height    = static_cast<float>(length) / width / threads_per_transform;
            int   cumheight = product(factors.begin(), factors.begin() + npass);

            int iheight = std::floor(height);
            if(height > iheight && threads_per_transform > length / width)
                iheight += 1;
            // std::cout << "iheight " << iheight << std::endl;
            if(npass != factors.size() - 1)
                for(int h = 0; h < iheight; ++h)
                    //work += generator(h, 0, width, 0);
                    for(int threadIdx = transform_id * threads_per_transform;
                        threadIdx < (transform_id + 1) * threads_per_transform;
                        ++threadIdx)
                        store_lds(length,
                                  threads_per_transform,
                                  regular_offset_lds,
                                  threadIdx,
                                  h,
                                  width,
                                  0,
                                  cumheight,
                                  elem_bytes,
                                  bank_bytes,
                                  debug);

            if(npass != 0)
                for(int h = 0; h < iheight; ++h)
                    //work += generator(h, 0, width, 0);
                    for(int threadIdx = transform_id * threads_per_transform;
                        threadIdx < (transform_id + 1) * threads_per_transform;
                        ++threadIdx)
                        load_lds(length,
                                 threads_per_transform,
                                 regular_offset_lds,
                                 threadIdx,
                                 h,
                                 width,
                                 0,
                                 elem_bytes,
                                 bank_bytes,
                                 debug);
        }
    }
}

int main(int argc, char* argv[])
{
    std::string      precision;
    std::vector<int> factors;

    int threads_per_transform;
    int wavefront_size;
    int bank_bytes;

    // clang-format off
    po::options_description opdesc("lds conflict sim options");
    opdesc.add_options()
        ("help,h", "produces this help message")
        ("precision,p",  po::value<std::string>(&precision)->default_value("float2"), "precision choices: float2, double2.")
        ("threads_per_transform,t", po::value<int>(&threads_per_transform)->default_value(4), "threads_per_transform")
        ("factors,f", po::value<std::vector<int>>(&factors)->multitoken(), "Radices to factorize the FFT.")
        ("wavefront_size,w", po::value<int>(&wavefront_size)->default_value(64), "wavefront_size")
        ("bank_bytes,b", po::value<int>(&bank_bytes)->default_value(4), "lds bank size in bytes.")
        ("verbose,v", "Print detailed debug info.");
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
            threads_per_transform, factors, wavefront_size, bank_bytes, vm.count("verbose"));
    else if(precision == "double2")
        st_batched_1d_lds_conflict_sim<double2>(
            threads_per_transform, factors, wavefront_size, bank_bytes, vm.count("verbose"));

    return 0;
}