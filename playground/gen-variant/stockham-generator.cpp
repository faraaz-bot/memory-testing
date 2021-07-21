
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
T product(std::vector<T> x, int last = -1)
{
    if(last == 0)
        return 1;
    if(last > 0)
        return std::accumulate(x.cbegin(), x.cbegin() + last, T(1), std::multiplies<T>());
    return std::accumulate(x.cbegin(), x.cend(), T(1), std::multiplies<T>());
}

struct StockhamGenerator
{
    // clang-format off
    Variable
          scalar_type{"scalar_type", "typename"}
        , write{"write", "bool"}
        , thread{"thread", "uint"}
        , thread_id{"thread_id", "void"}
        , lds{"lds", "scalar_type", .pointer=true, .restrict=true}
        , stride_lds{"stride_lds", "uint"}
        , offset_lds{"offset_lds", "uint"}
        , buf{"buf", "scalar_type", .pointer=true, .restrict=true}
        , stride_buf{"stride_buf", "size_t"}
        , offset_buf{"offset_buf", "size_t"}
        , twiddles{"twiddles", "scalar_type"}
        , R{"R", "scalar_type"}
        , W{"W", "scalar_type"}
        , t{"t", "scalar_type"}
        ;
    // clang-format on

    std::vector<int> factors;

    uint   length, width, nheight, threads_per_transform;
    double height;
    bool   half_lds;

    StockhamGenerator(std::vector<int> factors, uint threads_per_transform, bool half_lds = false)
        : factors(factors)
        , threads_per_transform(threads_per_transform)
        , half_lds(half_lds)
    {
        length = product(factors);

        uint nregisters = 0;
        for(auto width : factors)
        {
            uint n = ceil(double(length) / width / threads_per_transform) * width;
            if(n > nregisters)
                nregisters = n;
        }
        R.size = OptionalExpression(Literal{nregisters});
    };

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

    StatementList load_lds(uint h, uint width, int component)
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_lds + tid + w * (length / width);
            if(component < 0)
                stmts += Assign(R[h * width + w], lds[idx]);
            else if(component == 0)
                stmts += Assign(R[h * width + w].x, lds[idx].x);
            else if(component == 1)
                stmts += Assign(R[h * width + w].y, lds[idx].y);
        }
        return stmts;
    }

    StatementList load_global(uint h)
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_buf + (tid + w * (length / width)) * stride_buf;
            stmts += Assign(R[h * width + w], buf[idx]);
        }
        return stmts;
    }

    StatementList apply_twiddle(uint h)
    {
        StatementList stmts;
        for(uint w = 1; w < width; ++w)
        {
            auto tid  = thread + h * threads_per_transform;
            auto tidx = nheight - 1 + w - 1 + (width - 1) * (tid % nheight);
            auto ridx = h * width + w;
            stmts += Assign(W, twiddles[tidx]);
            stmts += Assign(t.x, W.x * R[ridx].x - W.y * R[ridx].y);
            stmts += Assign(t.y, W.y * R[ridx].x + W.x * R[ridx].y);
            stmts += Assign(R[ridx], t);
        }
        return stmts;
    }

    StatementList butterfly(uint h)
    {
        StatementList stmts;
        auto          args = ArgumentList();
        for(uint w = 0; w < width; ++w)
            args.append(R[h * width + w].address());
        stmts += Call("FwdRad" + std::to_string(width), args);
        return stmts;
    }

    StatementList store_lds(uint h, uint width, int component)
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_lds
                       + ((tid / nheight) * (width * nheight) + tid % nheight + w * nheight)
                             * stride_lds;
            if(component < 0)
                stmts += Assign(lds[idx], R[h * width + w]);
            else if(component == 0)
                stmts += Assign(lds[idx].x, R[h * width + w].x);
            else if(component == 1)
                stmts += Assign(lds[idx].y, R[h * width + w].y);
        }
        return stmts;
    }

    StatementList store_global(uint h)
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_buf
                       + ((tid / nheight) * (width * nheight) + tid % nheight + w * nheight)
                             * stride_buf;
            stmts += Assign(buf[idx], R[h * width + w]);
        }
        return stmts;
    }

    Function make_device()
    {
        auto kdevice = Function("forward_length" + std::to_string(length));

        kdevice.templates.append(scalar_type);

        kdevice.arguments.append(lds);
        kdevice.arguments.append(stride_lds);
        kdevice.arguments.append(offset_lds);
        if(half_lds)
        {
            kdevice.arguments.append(buf);
            kdevice.arguments.append(stride_buf);
            kdevice.arguments.append(offset_buf);
        }
        kdevice.arguments.append(write);

        kdevice.body += thread.declaration();
        kdevice.body += R.declaration();
        kdevice.body += W.declaration();
        kdevice.body += t.declaration();

        kdevice.body += Assign(thread, thread_id % threads_per_transform);

        for(uint pass = 0; pass < factors.size(); ++pass)
        {
            width   = factors[pass];
            height  = double(length) / width / threads_per_transform;
            nheight = product(factors, pass);

            kdevice.body += LineBreak();
            kdevice.body += CommentLines{
                "pass " + std::to_string(pass) + ", width " + std::to_string(width),
                "using " + std::to_string(threads_per_transform) + " threads we need to do "
                    + std::to_string(length / width) + " butterflies",
                "therefore each threads will do " + std::to_string(height) + " butterflies"};
            kdevice.body += SyncThreads();

            if(pass == 0 && half_lds)
                kdevice.body
                    += add_work([this](uint h) { return load_global(h); }, width, height, true);

            if (!half_lds)
                kdevice.body
                    += add_work([this](uint h) { return load_lds(h, width, -1); }, width, height);

            kdevice.body += add_work([this](uint h) { return apply_twiddle(h); }, width, height);
            kdevice.body += add_work([this](uint h) { return butterfly(h); }, width, height);

            if(half_lds)
            {
                if(pass < factors.size() - 1)
                {
                    for(uint component = 0; component < 2; ++component)
                    {
                        kdevice.body += add_work(
                            [&, this](uint h) { return store_lds(h, factors[pass], component); },
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
                        [this](uint h) { return store_global(h); }, width, height, true);
                }
            }
            else
            {
                kdevice.body += SyncThreads();
                kdevice.body += add_work(
                    [this](uint h) { return store_lds(h, width, -1); }, width, height, true);
            }
        }

        return kdevice;
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

    std::string cmd = "clang-format-10 -i -style=file " + tname;
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
    std::vector<int> factors;

    for(int i = 1; i < argc; ++i)
        factors.push_back(std::stoi(argv[i]));

    auto stockham = StockhamGenerator(factors, 7, false);
    auto device   = stockham.make_device();
    auto planar   = make_planar(device, "buf");

    format_and_write("stockham_generated_kernel.h", planar.render());
}
