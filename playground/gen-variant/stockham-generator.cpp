
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
    Variable R{"R"}, thread{"thread"}, thread_id{"thread_id"}, lds{"lds"}, offset_lds{"offset_lds"},
        write{"write"}, W{"W"}, t{"t"}, twiddles{"twiddles"}, lstride{"lstride"};

    std::vector<int> factors;

    uint   length, width, nheight, threads_per_transform;
    double height;

    StockhamGenerator(std::vector<int> factors)
        : factors(factors)
    {
        length = product(factors);
    };

    StatementList add_work(std::function<StatementList(uint)> generator, bool guard = false) const
    {
        auto iheight = floor(height);
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
            // XXX
        }

        return stmts;
    }

    StatementList load_lds(uint h)
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_lds + tid + w * (length / width);
            stmts += Assign(R[h * width + w], lds[idx]);
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
            args.arguments.push_back(R[h * width + w].address());
        stmts += Call("FwdRad" + std::to_string(width), args);
        return stmts;
    }

    StatementList store_lds(uint h)
    {
        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_lds + ((tid / nheight) * (width * nheight) + tid % nheight + w * nheight) * lstride;
            stmts += Assign(lds[idx], R[h * width + w]);
        }
        return stmts;
    }

    Function make_device()
    {
        threads_per_transform = length / factors[0];

        auto kdevice = Function("forward_length" + std::to_string(length));

        kdevice.body += R.declaration();
        kdevice.body += thread.declaration();
        kdevice.body += Assign(thread, thread_id % threads_per_transform);

        for(uint pass = 0; pass < factors.size(); ++pass)
        {
            width  = factors[pass];
            height = double(length) / width / threads_per_transform;
            nheight = product(factors, pass);

            kdevice.body += add_work([this](uint h) { return load_lds(h); });
            kdevice.body += add_work([this](uint h) { return apply_twiddle(h); });
            kdevice.body += add_work([this](uint h) { return butterfly(h); });
            kdevice.body += add_work([this](uint h) { return store_lds(h); });
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

    auto stockham = StockhamGenerator(factors);
    auto device   = stockham.make_device();

    format_and_write("stockham_generated_kernel.h", device.render());
}
