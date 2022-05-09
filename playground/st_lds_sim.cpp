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
//    ./st_lds_sim -t 8 -f 8 8 -m 1
//    ./st_lds_sim -t 7 -f 3 7 -m 1 -v
//    ./st_lds_sim -t 10 -f 5 5 4
//
// Notes:
//    - Typically, lds conflict becomes non-negligible for pow-of-2 cases.
//      There are few non-pow of 2 cases shown in this sim need to confirm.
//    - To minimize lds conflict, there might be a few solutions:
//      (a) lds padding based on fft length, (b) lds padding based simple bank
//      shift 16 or 32, (c) lda padding based on radix. Unlike solution of SBCC,
//      all these require more lds memory, in which might affect occupancy. (a)
//      seems doesn't help, while (c) seems too complicated introducing more ALU.
//      We only simulate (b) for now.
//    - Besides lds confilict, the total number of issued lds instructions is also
//      important.
//    - We are not simulating half lds loading directly yet.
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

// load_lds_generator
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
                      int  bank_shift,
                      bool debug)
{
    bank_ranges_t ret;
    auto          thread  = threadIdx % threads_per_transform;
    auto          lstride = 1;

    for(auto w = 0; w < width; ++w)
    {
        const auto tid = thread + dt + h * threads_per_transform;
        auto       idx = offset_lds + (tid + w * length / width) * lstride;

        if(bank_shift)
            idx = idx + idx / bank_shift;

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
            std::cout << "      tid " << std::setw(3) << threadIdx << " R: reg[" << std::setw(3)
                      << h * width + w << "], lds[" << std::setw(3) << idx << "], bank["
                      << std::setw(2) << bank_start << "-" << std::setw(2) << bank_end << "]"
                      << std::endl;
    }
    return ret;
}

// store_lds_generator
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
                      int  bank_shift,
                      bool debug)
{
    bank_ranges_t ret;
    auto          thread  = threadIdx % threads_per_transform;
    auto          lstride = 1;

    for(auto w = 0; w < width; ++w)
    {
        const auto tid = thread + dt + h * threads_per_transform;
        auto       idx
            = offset_lds
              + (tid / cumheight * (width * cumheight) + tid % cumheight + w * cumheight) * lstride;

        if(bank_shift)
            idx = idx + idx / bank_shift;

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
            std::cout << "      tid " << std::setw(3) << threadIdx << " W: lds[" << std::setw(3)
                      << idx << "], reg[" << std::setw(3) << h * width + w << "], bank["
                      << std::setw(2) << bank_start << "-" << std::setw(2) << bank_end << "]"
                      << std::endl;
    }
    return ret;
}

template <typename T>
void st_batched_1d_lds_conflict_sim(int               threads_per_transform,
                                    std::vector<int>& factors,
                                    float             max_transform_num,
                                    int               wavefront_size,
                                    int               bank_width,
                                    int               num_of_bank,
                                    int               bank_shift,
                                    bool              debug)
{
    const auto  length             = product(factors.begin(), factors.end());
    const auto  elem_bytes         = sizeof(T);
    const float transform_per_warp = (float)wavefront_size / threads_per_transform;

    std::cout << "-------------------------------------------------------------"
              << "\ntransform length:     " << length
              << "\nmax_transform_num:    " << max_transform_num
              << "\nthreads_per_transform:" << threads_per_transform
              << "\ntransform_per_warp:   " << transform_per_warp
              << "\nnum_of_bank:          " << num_of_bank
              << "\nbank_width:           " << bank_width
              << "\nwavefront_size:       " << wavefront_size << std::endl;

    for(auto group_id = 0; group_id < std::max(1, wavefront_size / num_of_bank); group_id++)
    {
        // The overall score, the lower the better
        int score = 0;

        // The idea target, hit bank maximum once per access.
        // * 2 for read and write per pass, except, no lds2reg at the 1st pass
        // and no reg2lds at the last pass.
        const int optimal_score = (std::accumulate(factors.begin(), factors.end(), 0) * 2
                                   - factors.front() - factors.back());

        auto group_start_thread = group_id * num_of_bank;
        auto group_end_thread
            = std::min((group_id + 1) * num_of_bank,
                       static_cast<int>(max_transform_num * threads_per_transform));

        if(group_start_thread < group_end_thread)
        {
            std::cout << "\nThread group " << group_id << ": [" << group_start_thread << ", "
                      << group_end_thread - 1 << "]" << std::endl;

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

                if(npass != factors.size() - 1)
                    for(auto threadIdx = group_start_thread; threadIdx < group_end_thread;
                        threadIdx++)
                    {
                        auto transform_id = threadIdx / threads_per_transform;

                        // no any padding or strides, elementwise
                        int regular_offset_lds = transform_id * length;

                        int   width  = factors[npass];
                        float height = static_cast<float>(length) / width / threads_per_transform;
                        int   cumheight = product(factors.begin(), factors.begin() + npass);

                        int iheight = std::floor(height);
                        if(height > iheight && threads_per_transform > length / width)
                            iheight += 1;
                        // std::cout << "iheight " << iheight << std::endl;

                        //stmts += CommentLines{"more than enough threads, some do nothing"};
                        //stmts += If{thread < length / width, work};
                        if((threads_per_transform == length / width)
                           || ((threads_per_transform != length / width)
                               && (threadIdx % threads_per_transform < length / width)))
                        {
                            if(debug)
                                std::cout << "    transform " << transform_id << std::endl;

                            for(auto h = 0; h < iheight; ++h) //work += generator(h, 0, width, 0);
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
                                                                    bank_shift,
                                                                    debug);
                                for(auto i = 0; i < bank_ranges.size(); ++i)
                                {
                                    for(auto b = bank_ranges[i].first; b <= bank_ranges[i].second;
                                        ++b)
                                    {
                                        write_hit_counts[i % width][b]++;
                                    }
                                }
                            }
                        }
                    }

                if(npass != 0)
                    for(auto threadIdx = group_start_thread; threadIdx < group_end_thread;
                        threadIdx++)
                    {
                        auto transform_id = threadIdx / threads_per_transform;

                        // no any padding or strides, elementwise
                        int regular_offset_lds = transform_id * length;

                        int   width  = factors[npass];
                        float height = static_cast<float>(length) / width / threads_per_transform;
                        int   cumheight = product(factors.begin(), factors.begin() + npass);

                        int iheight = std::floor(height);
                        if(height > iheight && threads_per_transform > length / width)
                            iheight += 1;

                        if((threads_per_transform == length / width)
                           || ((threads_per_transform != length / width)
                               && (threadIdx % threads_per_transform < length / width)))
                        {
                            if(debug)
                                std::cout << "    transform " << transform_id << std::endl;

                            for(auto h = 0; h < iheight; ++h) //work += generator(h, 0, width, 0);
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
                                                                    bank_shift,
                                                                    debug);
                                for(auto i = 0; i < bank_ranges.size(); ++i)
                                {
                                    for(auto b = bank_ranges[i].first; b <= bank_ranges[i].second;
                                        ++b)
                                    {
                                        read_hit_counts[i % width][b]++;
                                    }
                                }
                            }
                        }
                    }

                std::cout << "  R bank stats:\n  bank   ";
                for(auto i = 0; i < num_of_bank; i++)
                    std::cout << std::setw(2) << i << ",";
                std::cout << std::endl;
                for(auto w = 0; w < radix; ++w)
                {
                    auto max = read_hit_counts[w][0];
                    std::cout << "  step" << std::setw(2) << w << " ";
                    for(auto i = 0; i < num_of_bank; i++)
                    {
                        std::cout << std::setw(2) << read_hit_counts[w][i] << ",";
                        max = std::max(max, read_hit_counts[w][i]);
                    }
                    score += max;
                    std::cout << std::endl;
                }

                std::cout << "\n  W bank stats:\n  bank   ";
                for(auto i = 0; i < num_of_bank; i++)
                    std::cout << std::setw(2) << i << ",";
                std::cout << std::endl;
                for(auto w = 0; w < radix; ++w)
                {
                    auto max = write_hit_counts[w][0];
                    std::cout << "  step" << std::setw(2) << w << " ";
                    for(auto i = 0; i < num_of_bank; i++)
                    {
                        std::cout << std::setw(2) << write_hit_counts[w][i] << ",";
                        max = std::max(max, write_hit_counts[w][i]);
                    }
                    score += max;
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

            std::cout << "  Overall score: " << score << " vs bottom_line " << optimal_score
                      << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    std::string      precision;
    std::vector<int> factors;

    int   threads_per_transform;
    float max_transform_num;
    int   wavefront_size;
    int   bank_width;
    int   num_of_bank;
    int   bank_shift;

    // clang-format off
    po::options_description opdesc("lds conflict sim options");
    opdesc.add_options()
        ("help,h", "produces this help message")
        ("precision,p",  po::value<std::string>(&precision)->default_value("float"), "precision choices: float2, double2, float, double.")
        ("threads_per_transform,t", po::value<int>(&threads_per_transform)->default_value(4), "threads_per_transform")
        ("factors,f", po::value<std::vector<int>>(&factors)->multitoken(), "Radices to factorize the FFT.")
        ("max_transform_num,m", po::value<float>(&max_transform_num)->default_value(-1), "max number of transforms.")
        ("wavefront_size,w", po::value<int>(&wavefront_size)->default_value(64), "wavefront_size")
        ("bank_width,b", po::value<int>(&bank_width)->default_value(4), "lds physical bank width in bytes.")
        ("num_of_bank,n", po::value<int>(&num_of_bank)->default_value(32), "number of lds physical banks.")
        ("bank_shift,n", po::value<int>(&num_of_bank)->default_value(0), "shift 1 element in lds per bank_shift.")
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

    if(max_transform_num == -1)
        max_transform_num = static_cast<float>(wavefront_size) / threads_per_transform;

    if(precision == "float")
        st_batched_1d_lds_conflict_sim<float>(threads_per_transform,
                                              factors,
                                              max_transform_num,
                                              wavefront_size,
                                              bank_width,
                                              num_of_bank,
                                              bank_shift,
                                              vm.count("verbose"));
    else if(precision == "double")
        st_batched_1d_lds_conflict_sim<double>(threads_per_transform,
                                               factors,
                                               max_transform_num,
                                               wavefront_size,
                                               bank_width,
                                               num_of_bank,
                                               bank_shift,
                                               vm.count("verbose"));
    if(precision == "float2")
        st_batched_1d_lds_conflict_sim<float2>(threads_per_transform,
                                               factors,
                                               max_transform_num,
                                               wavefront_size,
                                               bank_width,
                                               num_of_bank,
                                               bank_shift,
                                               vm.count("verbose"));
    else if(precision == "double2")
        st_batched_1d_lds_conflict_sim<double2>(threads_per_transform,
                                                factors,
                                                max_transform_num,
                                                wavefront_size,
                                                bank_width,
                                                num_of_bank,
                                                bank_shift,
                                                vm.count("verbose"));

    return 0;
}
