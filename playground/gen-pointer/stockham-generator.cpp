//
// Simple AST based code generator for Stockham kernels.
//

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "generator.hpp"

using namespace gen;

//
// Stockham FFT.
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

    std::shared_ptr<ScalarVariable> scalar_type;
    std::shared_ptr<ScalarVariable> write, thread, thread_id;
    std::shared_ptr<ArrayVariable>  lds, buf, twiddles, R;
    std::shared_ptr<ScalarVariable> stride_lds, offset_lds;
    std::shared_ptr<ScalarVariable> stride_buf, offset_buf;
    std::shared_ptr<ScalarVariable> W, t;

    std::vector<int> factors;

    uint   length, width, nheight, threads_per_transform;
    double height;
    bool   half_lds;

    StockhamGenerator(std::vector<int>& factors, uint threads_per_transform, bool half_lds = false)
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

        scalar_type = scalar("scalar_type", "typename");
        write       = scalar("write", "bool");
        thread      = scalar("thread", "size_t");
        thread_id   = scalar("threadIdx.x", "");
        lds         = array("lds", "scalar_type");
        buf         = array("buf", "scalar_type");
        twiddles    = array("twiddles", "scalar_type");
        R           = array("R", "scalar_type", literal(nregisters));
        stride_lds  = scalar("stride_lds", "uint");
        offset_lds  = scalar("offset_lds", "uint");
        stride_buf  = scalar("stride_buf", "uint");
        offset_buf  = scalar("offset_buf", "uint");
        W           = scalar("W", "scalar_type");
        t           = scalar("t", "scalar_type");
    };

    StatementList add_work(std::function<StatementList(uint)> generator,
                           uint                               width,
                           double                             height,
                           bool guard = false) const
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
                stmts += if_block(write && (thread < length / width), work);
            else
                stmts += if_block(write, work);
        }
        else
        {
            stmts += work;
        }

        if(height > iheight && threads_per_transform < length / width)
        {
            work = generator(iheight);
            stmts += if_block(write && (thread + iheight * threads_per_transform < length / width),
                              work);
        }

        return stmts;
    }

    StatementList load_lds(uint h, uint width, int component)
    {
        auto R   = *std::dynamic_pointer_cast<ArrayVariable>(this->R);
        auto lds = *std::dynamic_pointer_cast<ArrayVariable>(this->lds);

        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_lds + tid + w * (length / width);
            if(component < 0)
                stmts += assign(R[h * width + w], lds[idx]);
            else if(component == 0)
                stmts += assign(R[h * width + w]->x, lds[idx]->x);
            else if(component == 1)
                stmts += assign(R[h * width + w]->y, lds[idx]->y);
        }
        return stmts;
    }

    StatementList load_global(uint h)
    {
        auto R   = *this->R;
        auto buf = *this->buf;

        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_buf + (tid + w * (length / width)) * stride_buf;
            stmts += assign(R[h * width + w], buf[idx]);
        }
        return stmts;
    }

    StatementList apply_twiddle(uint h)
    {
        auto R        = *this->R;
        auto twiddles = *this->twiddles;

        StatementList stmts;
        for(uint w = 1; w < width; ++w)
        {
            auto tid  = thread + h * threads_per_transform;
            auto tidx = (tid % nheight) * (width - 1) + (nheight - 1 + w - 1);
            auto ridx = h * width + w;
            stmts += assign(W, twiddles[tidx]);
            stmts += assign(t->x, W->x * R[ridx]->x - W->y * R[ridx]->y);
            stmts += assign(t->y, W->y * R[ridx]->x + W->x * R[ridx]->y);
            stmts += assign(R[ridx], t);
        }
        return stmts;
    }

    StatementList butterfly(uint h)
    {
        auto R = *this->R;

        StatementList stmts;
        auto          args = ArgumentList();
        for(uint w = 0; w < width; ++w)
            args.append(R[h * width + w]->address());
        stmts += call("FwdRad" + std::to_string(width), args);
        return stmts;
    }

    StatementList store_lds(uint h, uint width, int component)
    {
        /* if(lwidth < 0) */
        /*     lwidth = this->width; */
        auto R   = *this->R;
        auto lds = *this->lds;

        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_lds
                       + ((tid / nheight) * (width * nheight) + tid % nheight + w * nheight)
                             * stride_lds;
            if(component < 0)
                stmts += assign(lds[idx], R[h * width + w]);
            else if(component == 0)
                stmts += assign(lds[idx]->x, R[h * width + w]->x);
            else if(component == 1)
                stmts += assign(lds[idx]->y, R[h * width + w]->y);
        }
        return stmts;
    }

    StatementList store_global(uint h)
    {
        auto R   = *this->R;
        auto buf = *this->buf;

        StatementList stmts;
        for(uint w = 0; w < width; ++w)
        {
            auto tid = thread + h * threads_per_transform;
            auto idx = offset_buf
                       + ((tid / nheight) * (width * nheight) + tid % nheight + w * nheight)
                             * stride_buf;
            stmts += assign(buf[idx], R[h * width + w]);
        }
        return stmts;
    }

    std::shared_ptr<Function> make_device()
    {
        auto kdevice = std::make_shared<Function>("forward_length" + std::to_string(length));

        kdevice->templates.append(scalar_type);

        kdevice->arguments.append(lds);
        kdevice->arguments.append(stride_lds);
        kdevice->arguments.append(offset_lds);
        if(half_lds)
        {
            kdevice->arguments.append(buf);
            kdevice->arguments.append(stride_buf);
            kdevice->arguments.append(offset_buf);
        }
        kdevice->arguments.append(write);

        kdevice->body += thread->declaration();
        kdevice->body += R->declaration();
        kdevice->body += W->declaration();
        kdevice->body += t->declaration();

        kdevice->body += assign(thread, thread_id % threads_per_transform);

        for(uint pass = 0; pass < factors.size(); ++pass)
        {
            width   = factors[pass];
            height  = double(length) / width / threads_per_transform;
            nheight = product(factors, pass);

            kdevice->body += line_break();
            // kdevice.body += CommentLines{
            //     "pass " + std::to_string(pass) + ", width " + std::to_string(width),
            //     "using " + std::to_string(threads_per_transform) + " threads we need to do "
            //         + std::to_string(length / width) + " butterflies",
            //     "therefore each threads will do " + std::to_string(height) + " butterflies"};
            kdevice->body += sync_threads();

            if(pass == 0 && half_lds)
                kdevice->body
                    += add_work([this](uint h) { return load_global(h); }, width, height, true);

            if (!half_lds)
                kdevice->body
                    += add_work([this](uint h) { return load_lds(h, width, -1); }, width, height);

            kdevice->body += add_work([this](uint h) { return apply_twiddle(h); }, width, height);
            kdevice->body += add_work([this](uint h) { return butterfly(h); }, width, height);

            if(half_lds)
            {
                if(pass < factors.size() - 1)
                {
                    for(uint component = 0; component < 2; ++component)
                    {
                        kdevice->body += add_work(
                            [&, this](uint h) { return store_lds(h, factors[pass], component); },
                            factors[pass],
                            double(length) / factors[pass] / threads_per_transform,
                            true);
                        kdevice->body += sync_threads();

                        kdevice->body += add_work(
                            [&, this](uint h) { return load_lds(h, factors[pass + 1], component); },
                            factors[pass + 1],
                            double(length) / factors[pass + 1] / threads_per_transform);
                        kdevice->body += sync_threads();
                    }
                }
                else
                {
                    kdevice->body += add_work(
                        [this](uint h) { return store_global(h); }, width, height, true);
                }
            }
            else
            {
                kdevice->body += sync_threads();
                kdevice->body += add_work(
                    [this](uint h) { return store_lds(h, width, -1); }, width, height, true);
            }
        }

        return kdevice;
    }
};

int main(int argc, char* argv[])
{
    std::vector<int> factors;

    for(int i = 1; i < argc; ++i)
        factors.push_back(std::stoi(argv[i]));

    auto stockham = StockhamGenerator(factors, 7, true);
    auto device   = stockham.make_device();
    auto planar   = make_planar(device, "buf");

    /*
    auto global = make_global_fft(factors);
    auto host   = make_host_fft(factors);
*/

    format_and_write("stockham_generated_kernel.h",
                     planar->render());    // + global->render() + host->render());
}
