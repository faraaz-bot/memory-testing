#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include <set>

#include "../../../eg/argv/CLI11.hpp"
#include "../src/membench.hpp"

// Params for entire execution
class test_config
{
public:
    static size_t                   ngpus;
    static int                      verbose;
    static std::vector<hipStream_t> streams;
    static generator                gen;
};

size_t                   test_config::ngpus   = 4;
int                      test_config::verbose = 0;
std::vector<hipStream_t> test_config::streams;
generator                test_config::gen = gen_random;

// Params varying between tests
template <class T, size_t length>
struct params
{
    using Type                = T;
    static constexpr size_t N = length;
};

enum transpose_type
{
    t_block,
    t_local,
    t_normal
};

// Signature of a benchmark function defined in membench.hpp
template <typename T>
using benchmark_fn = std::function<float(const benchmark_context&, gpubuf_vec<T>&, gpubuf_vec<T>&)>;

// Configurations of type/length to test
using Params = ::testing::Types<params<float, 8>,
                                params<float, 16>,
                                params<float, 32>,
                                params<float, 64>,
                                params<float, 128>,
                                params<float, 256>,
                                params<float, 512>,
                                params<float, 1024>,
                                params<float, 2048>,
                                params<float, 4096>,
                                params<float, 8192>,
                                params<float, 16384>,
                                params<double, 8>,
                                params<double, 16>,
                                params<double, 32>,
                                params<double, 64>,
                                params<double, 128>,
                                params<double, 256>,
                                params<double, 512>,
                                params<double, 1024>,
                                params<double, 2048>,
                                params<double, 4096>,
                                params<double, 8192>,
                                params<double, 16384>>;

template <class Params>
class MembenchTest : public ::testing::Test
{
protected:
    // void SetUp() override {}
    // void TearDown() override {}
private:
    // Verify correctness for any benchmark transpose type
    template <typename Tfloat>
    bool is_correct(size_t                     N,
                    size_t                     ngpus,
                    const std::vector<Tfloat>& input,
                    gpubuf_vec<Tfloat>&        dev_out,
                    transpose_type             type)
    {
        std::vector<Tfloat> assembled_out(dev_out.size());
        assemble_output_to_host(N, ngpus, dev_out.data(), assembled_out.data());
        std::vector<Tfloat> reference(dev_out.size());

        switch(type)
        {
        case t_block:
            host_copy(N, test_config::ngpus, input.data(), reference.data());
            break;
        case t_local:
            break; // TODO
        case t_normal:
            host_transpose(N, input.data(), reference.data());
            break;
        }
        return is_same_matrix(N, reference, assembled_out);
    }

    template <typename Tfloat>
    void copy_host_buf_to_dev(std::vector<Tfloat>& h_bufs, gpubuf_vec<Tfloat>& d_bufs, size_t N)
    {
        const size_t buf_elems = N * N / test_config::ngpus;

        for(auto i = 0; i < test_config::ngpus; i++)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipMemcpy(d_bufs[i],
                                h_bufs.data() + i * buf_elems,
                                buf_elems * sizeof(Tfloat),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_bufs[i], 0, sizeof(Tfloat) * buf_elems));
        }
    }

public:
    using params = Params;

    // Execute arbitrary benchmark type
    template <typename Tfloat, transpose_type t_type>
    bool run_benchmark(size_t N, benchmark_fn<Tfloat> fn)
    {
        // Setup bufs
        std::vector<Tfloat> h_input = generate(
            N, N, test_config::gen, static_cast<Tfloat>(-100.f), static_cast<Tfloat>(100.f));
        gpubuf_vec<Tfloat> d_input(N, test_config::ngpus);
        gpubuf_vec<Tfloat> d_output(N, test_config::ngpus);
        copy_host_buf_to_dev(h_input, d_input, N);

        // std::string test_name = testing::UnitTest::GetInstance()->current_test_info()->name();
        return true;
    }
};

TYPED_TEST_SUITE(MembenchTest, Params);

TYPED_TEST(MembenchTest, Memcpy2D)
{
    size_t N   = TestFixture::params::N;
    using Type = typename TestFixture::params::Type;
    bool res   = this->template run_benchmark<Type, t_block>(N, run_memcpy<Type>);
    ASSERT_TRUE(res);
}

int main(int argc, char* argv[])
{
    // Note: also edit map in add_benchmarks() if editing this set
    std::set<std::string> valid_benchmarks = {"all",
                                              "memcpy2D",
                                              "memcpy2DAsync",
                                              "naiveCopy",
                                              "localTranspose",
                                              "naiveCopy+Transpose",
                                              "memcpy2D+Transpose",
                                              "memcpy2DAsync+Transpose"};
    // Parse args
    CLI::App app{"Memcpy test"};

    std::set<std::string> param_enabled_benchmarks;
    // app.add_option("-n, --length", ctx.N, "Length of input square matrix")->default_val(8U);
    app.add_option("-g, --ngpus", test_config::ngpus, "Number of gpus")
        ->default_val(4U)
        ->check(CLI::PositiveNumber);
    app.add_option("-v, --verbose",
                   test_config::verbose,
                   "Adjust output verbosity level\n1) Basic benchmark details\n2) Matrix data\n3) "
                   "Initial buffer data")
        ->default_val(0);

    app.add_option("-i, --inputGen",
                   test_config::gen,
                   "Data generation type:\n0) random (default)\n1) ordered (linear sequence)")
        ->default_val(0);
    // double    min_val;
    // double    max_val;
    // app.add_option("--min", min_val, "Minimum value to use if generating random input")
    //     ->default_val(-1.0);
    // app.add_option("--max", max_val, "Maximum value to use if generating random input")
    //     ->default_val(1.0);

    app.allow_extras();
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    // Check if inputs are valid for benchmark
    if(!(is_power_of_two(test_config::ngpus)))
        throw std::runtime_error("ngpus should be a power of two");

    test_config::streams.resize(test_config::ngpus * test_config::ngpus);
    // Enable peer to peer memory access between GPUs
    for(size_t i = 0; i < test_config::ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        for(size_t j = 0; j < test_config::ngpus; j++)
        {
            int can_access_peer;
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, i, j));
            if(can_access_peer)
                HIP_CHECK(hipDeviceEnablePeerAccess(j, 0));
        }
        // Also just setup some streams here
        for(size_t j = 0; j < test_config::ngpus; j++)
            HIP_CHECK(hipStreamCreate(&test_config::streams[i * test_config::ngpus + j]));
    }

    testing::InitGoogleTest(&argc, argv);
    auto retval = RUN_ALL_TESTS();
    return retval;
}
