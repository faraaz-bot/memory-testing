//
// Simple AST based code generator for Stockham kernels.
//

#include <memory>
#include <vector>

#include "generator.hpp"

using namespace gen;

//
// FFTs!!!
//

std::vector<int> unique_factors(std::vector<int> const& factors)
{
    auto result(factors);
    std::sort(result.begin(), result.end());
    auto end = std::unique(result.begin(), result.end());
    result.resize(std::distance(result.begin(), end));
    return result;
}

template <typename T>
T product(std::vector<T> x, int last = -1)
{
    if(last == 0)
        return 1;
    if(last > 0)
        return std::accumulate(x.cbegin(), x.cbegin() + last, T(1), std::multiplies<T>());
    return std::accumulate(x.cbegin(), x.cend(), T(1), std::multiplies<T>());
}

//
// Stockham pass
//
std::shared_ptr<Function> make_device_fft_pass(int pass, std::vector<int> factors)
{
    //
    // function and argument definitions
    //
    auto length = product(factors);
    auto fft = function("forward_length" + std::to_string(length) + "_pass" + std::to_string(pass));
    auto scalar_type = variable("scalar_type", "typename");
    auto input       = array("input", "scalar_type *");
    auto output      = array("output", "scalar_type *");
    auto rw          = variable("rw", "unsigned int");
    auto thread      = variable("thread", "unsigned int");
    auto twiddles    = array("twiddles", "const scalar_type *");
    auto stride_in   = variable("stride_in", "const size_t");
    auto stride_out  = variable("stride_out", "const size_t");
    auto offset_in   = variable("offset_in", "unsigned int");
    auto offset_out  = variable("offset_out", "unsigned int");

    fft->templates.push_back(scalar_type->argument());
    fft->arguments.push_back(input->argument());
    fft->arguments.push_back(output->argument());
    fft->arguments.push_back(rw->argument());
    fft->arguments.push_back(thread->argument());
    fft->arguments.push_back(twiddles->argument());
    fft->arguments.push_back(stride_in->argument());
    fft->arguments.push_back(stride_out->argument());
    fft->arguments.push_back(offset_in->argument());
    fft->arguments.push_back(offset_out->argument());

    //
    // register definitions
    //
    auto unique  = unique_factors(factors);
    auto width   = factors[pass];
    auto height  = product(unique) / width;
    auto nheight = product(factors, pass);

    for(int r = 0; r < width * height; ++r)
    {
        auto R = variable("R" + std::to_string(r), "scalar_type *");
        fft->arguments.push_back(R->argument());
    }

    std::vector<std::shared_ptr<ScalarVariable>> R(width * height);
    for(int r = 0; r < width * height; ++r)
        R[r] = scalar("(*R" + std::to_string(r) + ")");

    auto x = *input;
    auto z = *output;
    auto T = *twiddles;

    //
    // load
    //
    if(pass == 0)
    {
        auto load = if_block(rw);
        for(int w = 0; w < width; ++w)
        {
            for(int h = 0; h < height; ++h)
            {
                // clang-format off
                auto idx = add({
                    offset_in,
                    multiply({
                        group(add({
                              multiply({literal(height), thread}),
                              literal((length / width) * w + h)})),
                        stride_in
                      })
                  });
                // clang-format on
                load->body.push_back(assign(R[width * h + w], x[idx]));
            }
        }
        fft->body.push_back(load);
    }

    //
    // twiddle
    //
    if(pass > 0)
    {
        auto W  = variable("W", "scalar_type");
        auto Wx = variable("W.x", "scalar_type");
        auto Wy = variable("W.y", "scalar_type");
        auto t  = variable("t", "scalar_type");
        auto tx = variable("t.x", "scalar_type");
        auto ty = variable("t.y", "scalar_type");
        fft->body.push_back(W->declaration());
        fft->body.push_back(t->declaration());

        for(int h = 0; h < height; ++h)
        {
            for(int w = 1; w < width; ++w)
            {
                // clang-format off
                auto idx =
                  add({
                      literal(nheight - 1 + w - 1),
                      multiply({
                          literal(width - 1),
                          group(
                                mod({
                                    group(
                                          add({
                                              multiply({literal(height), thread}),
                                              literal(h)})),
                                    literal(nheight)}))})});
                // clang-format on
                auto ridx = h * width + w;
                auto Rx   = variable(R[ridx]->name + ".x", "");
                auto Ry   = variable(R[ridx]->name + ".y", "");
                fft->body.push_back(assign(W, T[idx]));
                fft->body.push_back(assign(tx, sub({multiply({Wx, Rx}), multiply({Wy, Ry})})));
                fft->body.push_back(assign(ty, add({multiply({Wy, Rx}), multiply({Wx, Ry})})));
                fft->body.push_back(assign(R[ridx], t));
            }
        }
    }

    //
    // butterflies
    //
    for(int h = 0; h < height; ++h)
    {
        auto fwd = call("FwdRad" + std::to_string(width) + "B1");
        for(int w = 0; w < width; ++w)
            fwd->arguments.push_back(scalar("R" + std::to_string(h * width + w)));
        fft->body.push_back(fwd);
    }

    //
    // store
    //
    auto store = if_block(rw);
    if(pass < factors.size() - 1)
    {
        for(int h = 0; h < height; ++h)
        {
            for(int w = 0; w < width; ++w)
            {
                // clang-format off
                auto base = group(add({multiply({literal(height), thread}), literal(h)}));
                auto idx = add({
                    offset_out,
                    multiply({
                        group(add({
                              multiply({group(divide({base, literal(nheight)})), literal(width*nheight)}),
                              mod({base, literal(nheight)}),
                              literal(w*nheight)})),
                        stride_out
                      })
                  });
                // clang-format on
                store->body.push_back(assign(z[idx], R[h * width + w]));
            }
        }
    }
    else
    {
        height = factors[0];
        width  = product(unique) / height;

        for(int w = 0; w < width; ++w)
        {
            for(int h = 0; h < height; ++h)
            {
                // clang-format off
                auto idx = add({
                    offset_out,
                    multiply({
                        group(add({
                              multiply({literal(height), thread}),
                              literal((length / width) * w + h)})),
                        stride_out
                      })});
                // clang-format on
                store->body.push_back(assign(z[idx], R[width * h + w]));
            }
        }
    }
    fft->body.push_back(store);

    //
    // reload
    //
    // XXX something funky when square
    if(pass < factors.size() - 1)
    {
        height = factors[0];
        width  = product(unique) / height;
        //        std::swap(width, height);

        auto reload = if_block(rw);
        for(int w = 0; w < width; ++w)
        {
            for(int h = 0; h < height; ++h)
            {
                // clang-format off
                auto idx = add({
                    offset_out,
                    multiply({literal(height), thread}),
                    literal((length / width) * w + h)});
                // clang-format on
                reload->body.push_back(assign(R[h * width + w], z[idx]));
            }
        }
        fft->body.push_back(reload);
    }

    return fft;
}

std::shared_ptr<Function> make_device_fft(int length, std::vector<int> factors)
{

    auto fft = function("fwd_len" + std::to_string(length) + "_device");

    auto T          = variable("T", "typename");
    auto sb         = variable("sb", "StrideBin");
    auto sync       = variable("sync", "bool");
    auto twiddles   = variable("twiddles", "const T *");
    auto stride_in  = variable("stride_in", "const size_t");
    auto stride_out = variable("stride_out", "const size_t");
    auto rw         = variable("rw", "unsigned int");
    auto me         = variable("me", "unsigned int");
    auto ldsOffset  = variable("ldsOffset", "unsigned int");

    fft->templates.push_back(T->argument());
    fft->templates.push_back(sync->argument());
    fft->arguments.push_back(twiddles->argument());
    fft->arguments.push_back(stride_in->argument());
    fft->arguments.push_back(stride_out->argument());
    fft->arguments.push_back(rw->argument());
    fft->arguments.push_back(me->argument());
    fft->arguments.push_back(ldsOffset->argument());

    auto lwbIn  = variable("lwbIn", "T *");
    auto lwbOut = variable("lwbOut", "T *");
    auto lds    = variable("lds", "T *");

    fft->arguments.push_back(lwbIn->argument());
    fft->arguments.push_back(lwbOut->argument());
    fft->arguments.push_back(lds->argument());

    // XXX
    int width = 14;
    for(int i = 0; i < width; ++i)
    {
        auto R = variable("R" + std::to_string(i), "T");
        fft->body.push_back(R->declaration());
    }

    for(int pass = 0; pass < factors.size(); ++pass)
    {
        auto fwd = call("FwdPass" + std::to_string(pass) + "_len" + std::to_string(length));
        fwd->templates.push_back(T);
        fwd->templates.push_back(sb);
        fwd->templates.push_back(sync);
        fwd->arguments.push_back(twiddles);
        fwd->arguments.push_back(stride_in);
        fwd->arguments.push_back(stride_out);
        fwd->arguments.push_back(rw);
        fwd->arguments.push_back(me);

        std::shared_ptr<Node> offset_in  = ldsOffset;
        std::shared_ptr<Node> offset_out = ldsOffset;

        if(pass == 0)
            offset_in = literal(0);
        if(pass == factors.size() - 1)
            offset_out = literal(0);

        fwd->arguments.push_back(offset_in);
        fwd->arguments.push_back(offset_out);
        fwd->arguments.push_back(lds);
        fwd->arguments.push_back(lds);
        fwd->arguments.push_back(lds);
        fwd->arguments.push_back(lds);

        for(int i = 0; i < width; ++i)
        {
            fwd->arguments.push_back(scalar("&R" + std::to_string(i)));
        }

        fft->body.push_back(fwd);
    }

    return fft;
}

int main(int argc, char* argv[])
{
    //std::vector<int> factors = {4, 4, 4, 4};
    std::vector<int> factors = {7, 2, 2, 2};
    //std::vector<int> factors = {7, 8};
    //std::vector<int> factors = {5, 3};
    for(int pass = 0; pass < factors.size(); ++pass)
    {
        auto pass_kernel = make_device_fft_pass(pass, factors);
        std::cout << pass_kernel->render() << std::endl;
    }
}
