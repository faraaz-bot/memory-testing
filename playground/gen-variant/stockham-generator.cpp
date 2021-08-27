
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <unistd.h>

#include "generator.hpp"

//
// Test!
//

template <typename T>
T product(std::vector<T> x)
{
    return std::accumulate(x.cbegin(), x.cend(), T(1), std::multiplies<T>());
}

template <typename Titer>
typename Titer::value_type product(Titer begin, Titer end)
{
    return std::accumulate(
        begin, end, typename Titer::value_type(1), std::multiplies<typename Titer::value_type>());
}

// extra optional parameters for the generator
struct Params
{
    bool        half_lds              = false;
    uint        threads_per_transform = 0;
    std::string scheme{"CS_KERNEL_STOCKHAM"};

    bool use_3steps_large_twd_sp = false;
    bool use_3steps_large_twd_dp = false;
};

struct StockhamGenerator : public Params
{
    std::vector<uint> factors;

    unsigned int length;
    unsigned int threads_per_block;
    unsigned int threads_per_transform;
    unsigned int batches_per_block;
    unsigned int nregisters;

    //
    // templates
    //
    Variable scalar_type{"scalar_type", "typename"};
    Variable callback_type{"cbtype", "CallbackType"};
    Variable stride_type{"sb", "StrideBin"};
    Variable embedded_type{"ebtype", "EmbeddedType"};

    //
    // arguments
    //
    // global input/ouput buffer
    Variable buf{"buf", "scalar_type", true, true};

    // global twiddle table (stacked)
    Variable twiddles{"twiddles", "const scalar_type", true, true};

    // rank/dimension of transform
    Variable dim{"dim", "const size_t"};

    // transform lengths
    Variable lengths{"lengths", "const size_t", true, true};

    // input/output array strides
    Variable stride{"stride", "const size_t", true, true};

    // number of transforms/batches
    Variable nbatch{"nbatch", "const size_t"};

    // the number of padding at the end of each row in lds
    Variable lds_padding{"lds_padding", "const unsigned int"};

    // should the device function write to lds?
    Variable write{"write", "bool"};

    //
    // locals
    //
    // lds storage buffer
    Variable lds{"lds", "scalar_type", true, true};

    // hip thread block id
    Variable block_id{"blockIdx.x", "unsigned int"};

    // hip thread id
    Variable thread_id{"threadIdx.x", "unsigned int"};

    // thread within transform
    Variable thread{"thread", "size_t"};

    // global input/output buffer offset to current transform
    Variable offset{"offset", "size_t"};

    // lds buffer offset to current transform
    Variable offset_lds{"offset_lds", "unsigned int"};

    // current batch
    Variable batch{"batch", "size_t"};

    // current transform
    Variable transform{"transform", "size_t"};

    // stride between consecutive indexes
    Variable stride0{"stride0", "const size_t"};

    // stride between consecutive indexes in lds
    Variable stride_lds{"stride_lds", "size_t"};

    // usually in device: const size_t lstride = (sb == SB_UNIT) ? 1 : stride_lds;
    // with this definition, the compiler knows that "index * lstride" is trivial under SB_UNIT
    Variable lstride{"lstride", "const size_t"};

    // twiddle value during twiddle application
    Variable W{"W", "scalar_type"};

    // temporary register during twiddle application
    Variable t{"t", "scalar_type"};

    // butterfly registers
    Variable R{"R", "scalar_type", false, false};

    static const unsigned int LDS_BYTE_LIMIT    = 32 * 1024;
    static const unsigned int BYTES_PER_ELEMENT = 16;
    StockhamGenerator(std::vector<uint> factors, uint _threads_per_block, const Params params)
        : Params(params)
        , factors(factors)
        , length(product(factors))
        , threads_per_block(_threads_per_block)
    {
        auto bytes_per_batch = length * BYTES_PER_ELEMENT;

        if(threads_per_transform == 0)
        {
            threads_per_transform = 1;
            for(uint t = 2; t < length; ++t)
            {
                if(t > threads_per_block)
                    continue;
                if(length % t == 0)
                {
                    if(std::all_of(factors.begin(), factors.end(), [=](uint f) {
                           return (length / t) % f == 0;
                       }))
                        threads_per_transform = t;
                }
            }
        }

        batches_per_block = LDS_BYTE_LIMIT / bytes_per_batch;
        while(threads_per_transform * batches_per_block > threads_per_block)
            --batches_per_block;
        threads_per_block = threads_per_transform * batches_per_block;

        nregisters = compute_nregisters(length, factors, threads_per_transform);
        R.size     = Expression{nregisters};
    }

    static unsigned int compute_nregisters(unsigned int              length,
                                           std::vector<unsigned int> factors,
                                           unsigned int              threads_per_transform)
    {
        uint max_registers = 0;
        for(auto width : factors)
        {
            uint n = ceil(double(length) / width / threads_per_transform) * width;
            if(n > max_registers)
                max_registers = n;
        }
        return max_registers;
    }

    // virtual methods for tiling
    virtual std::string tiling_name() const = 0;
    enum GlobalLoadDestination
    {
        TO_REGISTERS,
        TO_LDS,
    };
    enum GlobalStoreSource
    {
        FROM_REGISTERS,
        FROM_LDS,
    };
    virtual StatementList load_global(uint h, uint width, GlobalLoadDestination dest)           = 0;
    virtual StatementList store_global(uint h, uint width, uint nheight, GlobalStoreSource src) = 0;

    StatementList add_work(std::function<StatementList(uint)> generator,
                           uint                               width,
                           double                             height,
                           bool                               guard = false) const
    {
        uint iheight = floor(height);
        if(height > iheight && threads_per_transform > length / width)
            iheight += 1;

        auto work = StatementList();
        for(uint h = 0; h < iheight; ++h)
            work += generator(h);

        auto stmts = StatementList();
        if(guard)
        {
            if(threads_per_transform != length / width)
                stmts += If(write && (thread < length / width), work);
            else
                stmts += If(write, work);
        }
        else
        {
            stmts += work;
        }

        if(height > iheight && threads_per_transform < length / width)
        {
            work = generator(iheight);
            stmts += If(write && (thread + iheight * threads_per_transform < length / width), work);
        }

        return stmts;
    }

    enum class Component
    {
        REAL,
        IMAG,
        BOTH,
    };
    StatementList load_lds(uint h, uint width, Component component)
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_lds + (tid + w * (length / width)) * lstride;
            if(component == Component::BOTH)
                stmts += Assign(R[h * width + w], lds[idx]);
            else if(component == Component::REAL)
                stmts += Assign(R[h * width + w].x, lds[idx].x);
            else if(component == Component::IMAG)
                stmts += Assign(R[h * width + w].y, lds[idx].y);
        }
        return stmts;
    }

    StatementList apply_twiddle(uint h, uint width, uint nheight)
    {
        StatementList stmts;
        for(uint w = 1; w < width; ++w)
        {
            auto tid  = thread + h * threads_per_transform;
            auto tidx = nheight - 1 + w - 1 + (width - 1) * (tid % nheight);
            auto ridx = h * width + w;
            stmts += Assign(W, twiddles[tidx]);
            stmts += Assign(t, TwiddleMultiply({W, R[ridx]}));
            stmts += Assign(R[ridx], t);
        }
        return stmts;
    }

    StatementList butterfly(uint h, uint width)
    {
        StatementList           stmts;
        std::vector<Expression> args;
        for(uint w = 0; w < width; ++w)
            args.push_back(R + (h * width + w));
        stmts += Call("FwdRad" + std::to_string(width) + "B1", args);
        return stmts;
    }

    StatementList store_lds(uint h, uint width, Component component, uint nheight)
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx
                = offset_lds
                  + ((tid / nheight) * (width * nheight) + tid % nheight + w * nheight) * lstride;
            if(component == Component::BOTH)
                stmts += Assign(lds[idx], R[h * width + w]);
            else if(component == Component::REAL)
                stmts += Assign(lds[idx].x, R[h * width + w].x);
            else if(component == Component::IMAG)
                stmts += Assign(lds[idx].y, R[h * width + w].y);
        }
        return stmts;
    }

    Function make_device()
    {
        Function kdevice("forward_length" + std::to_string(length) + "_" + tiling_name()
                         + "_device");

        kdevice.qualifier = "__device__";

        kdevice.templates.append(scalar_type);
        kdevice.templates.append(stride_type);

        kdevice.arguments.append(lds);
        kdevice.arguments.append(twiddles);
        kdevice.arguments.append(stride_lds);
        kdevice.arguments.append(offset_lds);
        if(half_lds)
        {
            kdevice.arguments.append(buf);
            kdevice.arguments.append(stride0);
            kdevice.arguments.append(offset);
        }
        kdevice.arguments.append(write);

        kdevice.body += Declaration(thread, thread_id % threads_per_transform);
        kdevice.body += Declaration(R);
        kdevice.body += Declaration(W);
        kdevice.body += Declaration(t);
        kdevice.body += Declaration(
            lstride, Ternary(stride_type == Literal{"SB_UNIT"}, 1, stride_lds));

        for(uint pass = 0; pass < factors.size(); ++pass)
        {
            auto width   = factors[pass];
            auto height  = double(length) / width / threads_per_transform;
            auto nheight = product(factors.begin(), factors.begin() + pass);

            kdevice.body += LineBreak();
            kdevice.body += CommentLines{
                "pass " + std::to_string(pass) + ", width " + std::to_string(width),
                "using " + std::to_string(threads_per_transform) + " threads we need to do "
                    + std::to_string(length / width) + " butterflies",
                "therefore each thread will do " + std::to_string(height) + " butterflies"};
            kdevice.body += SyncThreads();

            if(pass == 0 && half_lds)
                kdevice.body
                    += add_work([=](uint h) { return load_global(h, width, TO_REGISTERS); },
                                width,
                                height,
                                true);

            if(!half_lds)
                kdevice.body += add_work(
                    [=](uint h) { return load_lds(h, width, Component::BOTH); }, width, height);

            if(pass > 0)
            {
                kdevice.body += add_work(
                    [=](uint h) { return apply_twiddle(h, width, nheight); }, width, height);
            }
            kdevice.body += add_work([=](uint h) { return butterfly(h, width); }, width, height);

            if(half_lds)
            {
                if(pass < factors.size() - 1)
                {
                    for(auto component : {Component::REAL, Component::IMAG})
                    {
                        kdevice.body += add_work(
                            [&, this](uint h) {
                                return store_lds(h, factors[pass], component, nheight);
                            },
                            factors[pass],
                            double(length) / factors[pass] / threads_per_transform,
                            true);
                        kdevice.body += SyncThreads();

                        // XXX sync
                        kdevice.body += add_work(
                            [&, this](uint h) { return load_lds(h, factors[pass + 1], component); },
                            factors[pass + 1],
                            double(length) / factors[pass + 1] / threads_per_transform);
                        kdevice.body += SyncThreads();
                    }
                }
                else
                {
                    kdevice.body += add_work(
                        [=](uint h) { return store_global(h, width, nheight, FROM_REGISTERS); },
                        width,
                        height,
                        true);
                }
            }
            else
            {
                kdevice.body += SyncThreads();
                kdevice.body += add_work(
                    [=](uint h) { return store_lds(h, width, Component::BOTH, nheight); },
                    width,
                    height,
                    true);
            }
        }
        return kdevice;
    }

    Function make_global()
    {
        Function kglobal("forward_length" + std::to_string(length) + "_" + tiling_name());
        kglobal.qualifier     = "__global__";
        kglobal.launch_bounds = threads_per_block;

        kglobal.templates.append(scalar_type);
        kglobal.templates.append(stride_type);
        kglobal.templates.append(embedded_type);
        kglobal.templates.append(callback_type);

        kglobal.arguments.append(twiddles);
        kglobal.arguments.append(dim);
        kglobal.arguments.append(lengths);
        kglobal.arguments.append(stride);
        kglobal.arguments.append(nbatch);
        kglobal.arguments.append(lds_padding);
        add_callback_arguments(kglobal.arguments);
        kglobal.arguments.append(buf);

        kglobal.body += CommentLines{
            std::string("this kernel:"),
            "  uses " + std::to_string(threads_per_transform) + " threads per transform",
            "  does " + std::to_string(batches_per_block) + " transforms per thread block",
            "therefore it should be called with " + std::to_string(threads_per_block)
                + " threads per thread block"};

        kglobal.body += LDSDeclaration(scalar_type.name);
        kglobal.body += Declaration(offset, 0);
        kglobal.body += Declaration(offset_lds);
        kglobal.body += Declaration(stride_lds);
        kglobal.body += Declaration(batch);
        kglobal.body += Declaration(transform);
        kglobal.body += Declaration(thread);
        kglobal.body += Declaration(write);
        kglobal.body += Declaration(
            stride0, Ternary{stride_type == Literal{"SB_UNIT"}, 1, stride[0]});
        kglobal.body += CallbackDeclaration(scalar_type.name, callback_type.name);

        kglobal.body += LineBreak();

        kglobal.body += CommentLines{std::string{"offsets"}};

        Variable remaining{"remaining", "size_t"};
        Variable index_along_d{"index_along_d", "size_t"};
        kglobal.body += Declaration{remaining};
        kglobal.body += Declaration{index_along_d};
        kglobal.body += Assign{
            transform, block_id * batches_per_block + thread_id / threads_per_transform};
        kglobal.body += Assign{remaining, transform};
        Variable d{"d", "int"};
        For      offset_for{d, 1, d < dim, 1};
        offset_for.body += Assign{index_along_d, remaining % lengths[d]};
        offset_for.body += Assign{remaining, remaining / lengths[d]};
        offset_for.body += Assign{offset, offset + index_along_d * stride[d]};
        kglobal.body += offset_for;

        kglobal.body += Assign{batch, remaining};
        kglobal.body += Assign{offset, offset + batch * stride[dim]};
        kglobal.body += Assign{
            offset_lds, (length + lds_padding) * (transform % batches_per_block)};

        kglobal.body += LineBreak();

        kglobal.body += If{batch >= nbatch, {Return()}};

        // FIXME: this should be pushed down to the RR-specific class?
        kglobal.body += CommentLines{std::string{"load global"}};
        kglobal.body += Assign{thread, thread_id % threads_per_transform};
        kglobal.body += add_work(
            [=](uint h) { return load_global(h, threads_per_transform, TO_LDS); }, 1, 1);

        kglobal.body += LineBreak();
        kglobal.body
            += CommentLines{std::string("append extra global loading for C2Real pre-process only")};
        StatementList c2real_pre;
        c2real_pre += CommentLines{
            "use the last thread of each transform to load one more element per row"};
        auto width  = threads_per_transform;
        auto height = length / width;
        c2real_pre += If{
            thread == threads_per_transform - 1,
            {Assign{lds[offset_lds + thread + (height - 1) * width + 1],
                    LoadGlobal{buf, offset + (thread + (height - 1) * width + 1) * stride0}}}};
        kglobal.body += If{embedded_type == Literal{"EmbeddedType::C2Real_PRE"}, c2real_pre};

        kglobal.body += CommentLines{std::string{"transform"}};
        kglobal.body += Assign{write, Literal{"true"}};
        kglobal.body += Call{kglobal.name + "_device",
                             {scalar_type, Variable{"SB_UNIT", "StrideBin"}},
                             {lds, twiddles, stride_lds, offset_lds, write}};

        kglobal.body += LineBreak();
        kglobal.body += CommentLines{
            std::string("handle even-length complex to real post-process in lds after transform")};
        StatementList real2c_post;
        real2c_post += SyncThreads();
        // FIXME: loop if necessary, compute all the numbers and Ndiv4 properly
        real2c_post += Call{"real_post_process_kernel_inplace",
                            {scalar_type, Variable{"false", "bool"}},
                            {thread % 5 + 0,
                             9 - thread % 5 - 0,
                             5,
                             lds + offset_lds,
                             0,
                             twiddles + 9}};
        kglobal.body += If{embedded_type == Literal{"EmbeddedType::Real2C_POST"}, real2c_post};

        kglobal.body += CommentLines{std::string{"store global"}};
        kglobal.body += SyncThreads();
        kglobal.body += add_work(
            [=](uint h) { return store_global(h, threads_per_transform, 1, FROM_LDS); }, 1, 1);

        return kglobal;
    }

    void add_callback_arguments(ArgumentList& args)
    {
        args.append(Variable{"load_cb_fn", "void", true, true});
        args.append(Variable{"load_cb_data", "void", true, true});
        args.append(Variable{"load_cb_lds_bytes", "uint32_t"});
        args.append(Variable{"store_cb_fn", "void", true, true});
        args.append(Variable{"store_cb_data", "void", true, true});
    }
};

struct StockhamGeneratorSBRR : public StockhamGenerator
{
    StockhamGeneratorSBRR(std::vector<uint> factors, uint threads_per_block, const Params params)
        : StockhamGenerator(factors, threads_per_block, params)
    {
    }

    std::string tiling_name() const override
    {
        return "SBRR";
    }

    StatementList load_global(uint h, uint width, GlobalLoadDestination dest) override
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset + (tid + w * (length / width)) * stride0;
            if(dest == TO_REGISTERS)
                stmts += Assign(R[h * width + w], LoadGlobal(buf, idx));
            else
                stmts
                    += Assign(lds[offset_lds + thread + w * width], LoadGlobal(buf, idx));
        }
        return stmts;
    }

    StatementList store_global(uint h, uint width, uint nheight, GlobalStoreSource src) override
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx
                = offset
                  + ((tid / nheight) * (width * nheight) + tid % nheight + w * nheight) * stride0;
            if(src == FROM_REGISTERS)
                stmts += StoreGlobal(buf, idx, R[h * width + w]);
            else
                stmts += StoreGlobal(buf, idx, lds[offset_lds + thread + w * width]);
        }
        return stmts;
    }
};

void format_and_write(std::string fname, std::string code)
{
    std::ofstream     ofile;
    std::ifstream     ifile;
    std::stringstream formatted, existing;

    auto tname = fname + ".tmp";

    ofile.open(tname);
    ofile << code;
    ofile.close();

    std::string cmd = "/opt/rocm/llvm/bin/clang-format -i -style=file " + tname;
    std::system(cmd.c_str());
    ifile.open(tname);
    formatted << ifile.rdbuf();
    ifile.close();

    unlink(tname.c_str());

    bool exists = static_cast<bool>(std::ifstream(fname));
    if(exists)
    {
        ifile.open(fname);
        existing << ifile.rdbuf();
        ifile.close();

        if(formatted.str().compare(existing.str()) == 0)
            return;
    }

    ofile.open(fname);
    ofile << formatted.str();
    ofile.close();
}

int main(int argc, char* argv[])
{
    std::vector<uint> factors;

    for(int i = 1; i < argc; ++i)
        factors.push_back(std::stoi(argv[i]));

    auto stockham = StockhamGeneratorSBRR(factors, 256, {});
    auto device   = stockham.make_device();
    auto global   = stockham.make_global();

    auto planar_global = make_planar(global, "buf");

    auto op_global = make_outofplace(global);

    auto ip_global = make_inplace(global);

    auto inverse_global = make_inverse(global);
    auto inverse_device = make_inverse(device);

    std::string rtc_typedefs = "typedef float2 scalar_type;"
                               "static const StrideBin sb = SB_UNIT;"
                               "static const EmbeddedType ebtype = EmbeddedType::NONE;"
                               "static const CallbackType cbtype = CallbackType::NONE;";
    auto ip_rtc = make_rtc(ip_global);

    format_and_write("stockham_generated_kernel.h",
                     device.render() + planar_global.render() + ip_global.render()
                         + op_global.render() + inverse_device.render() + inverse_global.render()
                         + rtc_typedefs + ip_rtc.render());
}
