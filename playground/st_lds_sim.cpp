///////////////////////////////////////////////////////////////////////////////
//
// Description:
//    To simulate lds conflict of 1D batched Stockham for all intermediate
//    passes(assume reg -> lds -> ... -> lds -> reg).
//    The elements swapping happens between two passes: reg2lds then lds2reg.
//
// Build:
//    hipcc st_lds_sim.cpp -o st_lds_sim -lboost_program_options
//
// Quick tests:
//    ./st_lds_sim -t 2 -f 4 2 -w 2 -v
//    ./st_lds_sim -t 2 -f 4 2 -w 4 -v
//    ./st_lds_sim -t 8 -f 8 8
//    ./st_lds_sim -t 10 -f 5 5 4
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/program_options.hpp>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace po = boost::program_options;

typedef std::vector<std::pair<int, int>> bank_ranges_t;

template <typename Titer>
typename Titer::value_type product(Titer begin, Titer end)
{
    return std::accumulate(
        begin, end, typename Titer::value_type(1), std::multiplies<typename Titer::value_type>());
}

bank_ranges_t lds2reg(int  length,
                      int  threads_per_transform,
                      int  offset_lds,
                      int  threadIdx,
                      int  h,
                      int  width,
                      int  dt,
                      int  elem_bytes,
                      int  bank_width,
                      int  num_of_bank,
                      bool debug)
{
    bank_ranges_t ret;
    auto          thread  = threadIdx % threads_per_transform;
    auto          lstride = 1;

    for(auto w = 0; w < width; ++w)
    {
        const auto tid = thread + dt + h * threads_per_transform;
        const auto idx = offset_lds + (tid + w * length / width) * lstride;

        const auto bank_start = idx * elem_bytes / bank_width % num_of_bank;
        const auto bank_end   = ((idx + 1) * elem_bytes - 1) / bank_width % num_of_bank;
        ret.push_back(std::make_pair(bank_start, bank_end));
        // switch(elem_bytes)
        // {
        // case 4:
        //     ret.push_back(std::make_tuple(bank_start, 0, 0, 0));
        //     break;
        // case 8:
        //     ret.push_back(std::make_tuple(bank_start, bank_start + 1, 0, 0));
        //     break;
        // case 12:
        //     ret.push_back(std::make_tuple(bank_start, bank_start + 1, bank_start + 2, 0));
        //     break;
        // case 16:
        //     ret.push_back(
        //         std::make_tuple(bank_start, bank_start + 1, bank_start + 2, bank_start + 3));
        //     break;
        // default:
        //     break;
        // }

        if(debug)
            std::cout << "    tid " << std::setw(3) << threadIdx << " R: reg[" << std::setw(3)
                      << h * width + w << "], lds[" << std::setw(3) << idx << "], bank["
                      << std::setw(2) << bank_start << "-" << std::setw(2) << bank_end << "]"
                      << std::endl;
    }
    return ret;
}

bank_ranges_t reg2lds(int  length,
                      int  threads_per_transform,
                      int  offset_lds,
                      int  threadIdx,
                      int  h,
                      int  width,
                      int  dt,
                      int  cumheight,
                      int  elem_bytes,
                      int  bank_width,
                      int  num_of_bank,
                      bool debug)
{
    bank_ranges_t ret;
    auto          thread  = threadIdx % threads_per_transform;
    auto          lstride = 1;

    for(auto w = 0; w < width; ++w)
    {
        const auto tid = thread + dt + h * threads_per_transform;
        const auto idx
            = offset_lds
              + (tid / cumheight * (width * cumheight) + tid % cumheight + w * cumheight) * lstride;

        const auto bank_start = idx * elem_bytes / bank_width % num_of_bank;
        const auto bank_end   = ((idx + 1) * elem_bytes - 1) / bank_width % num_of_bank;
        ret.push_back(std::make_pair(bank_start, bank_end));
        // switch(elem_bytes)
        // {
        // case 4:
        //     ret.push_back(std::make_tuple(bank_start, 0, 0, 0));
        //     break;
        // case 8:
        //     ret.push_back(std::make_tuple(bank_start, bank_start + 1, 0, 0));
        //     break;
        // case 12:
        //     ret.push_back(std::make_tuple(bank_start, bank_start + 1, bank_start + 2, 0));
        //     break;
        // case 16:
        //     ret.push_back(
        //         std::make_tuple(bank_start, bank_start + 1, bank_start + 2, bank_start + 3));
        //     break;
        // default:
        //     break;
        // }
        if(debug)
            std::cout << "    tid " << std::setw(3) << threadIdx << " W: lds[" << std::setw(3)
                      << idx << "], reg[" << std::setw(3) << h * width + w << "], bank["
                      << std::setw(2) << bank_start << "-" << std::setw(2) << bank_end << "]"
                      << std::endl;
    }
    return ret;
}

template <typename T>
void st_batched_1d_lds_conflict_sim(int               threads_per_transform,
                                    std::vector<int>& factors,
                                    int               wavefront_size,
                                    int               bank_width,
                                    int               num_of_bank,
                                    bool              debug)
{
    std::cout << "bank_width: " << bank_width << ", num_of_bank " << num_of_bank << std::endl;

    const auto  length             = product(factors.begin(), factors.end());
    const auto  elem_bytes         = sizeof(T);
    const float transform_per_warp = (float)wavefront_size / threads_per_transform;
    std::cout << "transform_per_warp: " << transform_per_warp << std::endl;

    // The overall score, the lower the better
    int score = 0;

    // The idea target, hit bank once per access.
    // * 2 for read and write per pass, except, no lds2reg at the 1st pass
    // and no reg2lds at the last pass.
    const int optimal_score = (std::accumulate(factors.begin(), factors.end(), 0) * 2
                               - factors.front() - factors.back())
                              * num_of_bank;

    for(auto npass = 0; npass < factors.size(); ++npass)
    {
        std::cout << "Pass " << npass << std::endl;

        const auto radix = factors[npass];

        // 2D bank stats storage for all steps in one pass
        int** read_hit_counts = new int*[radix];
        for(int i = 0; i < radix; i++)
        {
            read_hit_counts[i] = new int[num_of_bank];
        }
        int** write_hit_counts = new int*[radix];
        for(int i = 0; i < radix; i++)
        {
            write_hit_counts[i] = new int[num_of_bank];
        }
        for(int w = 0; w < radix; ++w)
        {
            for(auto i = 0; i < num_of_bank; i++)
                read_hit_counts[w][i] = write_hit_counts[w][i] = 0;
        }

        // simulate the first n complete transforms only for now
        for(int transform_id = 0; transform_id < transform_per_warp; transform_id++)
        {
            std::cout << "    transform " << transform_id << std::endl;
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
                for(auto h = 0; h < iheight; ++h)
                    //work += generator(h, 0, width, 0);
                    for(auto threadIdx = transform_id * threads_per_transform;
                        threadIdx < (transform_id + 1) * threads_per_transform;
                        ++threadIdx)
                    {
                        bank_ranges_t bank_ranges = reg2lds(length,
                                                            threads_per_transform,
                                                            regular_offset_lds,
                                                            threadIdx,
                                                            h,
                                                            width,
                                                            0,
                                                            cumheight,
                                                            elem_bytes,
                                                            bank_width,
                                                            num_of_bank,
                                                            debug);
                        for(auto bank_pair : bank_ranges)
                        {
                            for(auto w = 0; w < width; ++w)
                                for(auto i = bank_pair.first; i <= bank_pair.second; i++)
                                {
                                    write_hit_counts[w][i]++;
                                    score++;
                                }
                        }
                    }

            if(npass != 0)
                for(auto h = 0; h < iheight; ++h)
                    //work += generator(h, 0, width, 0);
                    for(auto threadIdx = transform_id * threads_per_transform;
                        threadIdx < (transform_id + 1) * threads_per_transform;
                        ++threadIdx)
                    {
                        bank_ranges_t bank_ranges = lds2reg(length,
                                                            threads_per_transform,
                                                            regular_offset_lds,
                                                            threadIdx,
                                                            h,
                                                            width,
                                                            0,
                                                            elem_bytes,
                                                            bank_width,
                                                            num_of_bank,
                                                            debug);
                        for(auto bank_pair : bank_ranges)
                        {
                            for(auto w = 0; w < width; ++w)
                                for(auto i = bank_pair.first; i <= bank_pair.second; i++)
                                {
                                    read_hit_counts[w][i]++;
                                    score++;
                                }
                        }
                    }
        }

        std::cout << "  W bank stats:\n  bank   ";
        for(auto i = 0; i < num_of_bank; i++)
            std::cout << std::setw(2) << i << ",";
        std::cout << std::endl;
        for(auto w = 0; w < radix; ++w)
        {
            std::cout << "  step" << std::setw(2) << w << " ";
            for(auto i = 0; i < num_of_bank; i++)
                std::cout << std::setw(2) << write_hit_counts[w][i] << ",";
            std::cout << std::endl;
        }

        std::cout << "\n  R bank stats:\n  bank   ";
        for(auto i = 0; i < num_of_bank; i++)
            std::cout << std::setw(2) << i << ",";
        std::cout << std::endl;
        for(auto w = 0; w < radix; ++w)
        {
            std::cout << "  step" << std::setw(2) << w << " ";
            for(auto i = 0; i < num_of_bank; i++)
                std::cout << std::setw(2) << read_hit_counts[w][i] << ",";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        for(int i = 0; i < radix; i++)
        {
            delete[] write_hit_counts[i];
        }
        delete[] write_hit_counts;
        for(int i = 0; i < radix; i++)
        {
            delete[] read_hit_counts[i];
        }
        delete[] read_hit_counts;
    }

    std::cout << "Overall score: " << score << " vs bottom_line " << optimal_score << std::endl;
}

int main(int argc, char* argv[])
{
    std::string      precision;
    std::vector<int> factors;

    int threads_per_transform;
    int wavefront_size;
    int bank_width;
    int num_of_bank;

    // clang-format off
    po::options_description opdesc("lds conflict sim options");
    opdesc.add_options()
        ("help,h", "produces this help message")
        ("precision,p",  po::value<std::string>(&precision)->default_value("float2"), "precision choices: float2, double2, float, double.")
        ("threads_per_transform,t", po::value<int>(&threads_per_transform)->default_value(4), "threads_per_transform")
        ("factors,f", po::value<std::vector<int>>(&factors)->multitoken(), "Radices to factorize the FFT.")
        ("wavefront_size,w", po::value<int>(&wavefront_size)->default_value(64), "wavefront_size")
        ("bank_width,b", po::value<int>(&bank_width)->default_value(4), "lds physical bank width in bytes.")
        ("num_of_bank,n", po::value<int>(&num_of_bank)->default_value(32), "number of lds physical banks.")
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

    if(precision == "float")
        st_batched_1d_lds_conflict_sim<float>(threads_per_transform,
                                              factors,
                                              wavefront_size,
                                              bank_width,
                                              num_of_bank,
                                              vm.count("verbose"));
    else if(precision == "double")
        st_batched_1d_lds_conflict_sim<double>(threads_per_transform,
                                               factors,
                                               wavefront_size,
                                               bank_width,
                                               num_of_bank,
                                               vm.count("verbose"));
    if(precision == "float2")
        st_batched_1d_lds_conflict_sim<float2>(threads_per_transform,
                                               factors,
                                               wavefront_size,
                                               bank_width,
                                               num_of_bank,
                                               vm.count("verbose"));
    else if(precision == "double2")
        st_batched_1d_lds_conflict_sim<double2>(threads_per_transform,
                                                factors,
                                                wavefront_size,
                                                bank_width,
                                                num_of_bank,
                                                vm.count("verbose"));

    return 0;
}
