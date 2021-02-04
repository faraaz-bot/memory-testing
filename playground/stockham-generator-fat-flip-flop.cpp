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
    auto inout       = array("inout", "scalar_type *");
    auto lds         = array("lds", "scalar_type *");
    auto registers   = array("R", "scalar_type *");
    auto rw          = variable("rw", "bool");
    auto thread      = variable("thread", "int");
    auto twiddles    = array("twiddles", "const scalar_type *");
    auto stride_in   = variable("stride_in", "int");
    auto stride_out  = variable("stride_out", "int");
    auto offset_in   = variable("offset_in", "int");
    auto offset_out  = variable("offset_out", "int");

    fft->templates.push_back(scalar_type->argument());
    fft->arguments.push_back(inout->argument());
    fft->arguments.push_back(lds->argument());
    fft->arguments.push_back(registers->argument());
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

    auto Z = *inout;
    auto X = *lds;
    auto R = *registers;
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
                load->body.push_back(assign(R[width * h + w], Z[idx]));
            }
        }
        fft->body.push_back(load);
    }

    //
    // twiddle
    //
    if(pass > 0)
    {
        auto W = scalar("W", "scalar_type");
        auto t = scalar("t", "scalar_type");
        fft->body.push_back(W->declaration());
        fft->body.push_back(t->declaration());

        for(int h = 0; h < height; ++h)
        {
            for(int w = 1; w < width; ++w)
            {
                // clang-format off
                auto tidx =
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
                fft->body.push_back(assign(W, T[tidx]));
                fft->body.push_back(assign(
                    t->x, sub({multiply({W->x, R[ridx]->x}), multiply({W->y, R[ridx]->y})})));
                fft->body.push_back(assign(
                    t->y, add({multiply({W->y, R[ridx]->x}), multiply({W->x, R[ridx]->y})})));
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
            fwd->arguments.push_back(R[h * width + w]->address());
        fft->body.push_back(fwd);
    }

    //
    // store
    //
    auto store = if_block(rw);
    if(pass < factors.size() - 1)
    {
        // write to lds
        for(int h = 0; h < height; ++h)
        {
            for(int w = 0; w < width; ++w)
            {
                // clang-format off
                auto base = group(add({multiply({literal(height), thread}), literal(h)}));
                auto idx = add({
                        offset_out,
                        group(add({
                              multiply({group(divide({base, literal(nheight)})), literal(width*nheight)}),
                              mod({base, literal(nheight)}),
                              literal(w*nheight)})),
                  });
                // clang-format on
                store->body.push_back(assign(X[idx], R[h * width + w]));
            }
        }
    }
    else
    {
        // write to global
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
                store->body.push_back(assign(Z[idx], R[width * h + w]));
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
                reload->body.push_back(assign(R[h * width + w], X[idx]));
            }
        }
        fft->body.push_back(reload);
    }

    return fft;
}

std::shared_ptr<Function> make_device_fft(std::vector<int> factors)
{
    //
    // function and argument definitions
    //
    auto length      = product(factors);
    auto fft         = function("forward_length" + std::to_string(length) + "_fat");
    auto scalar_type = variable("scalar_type", "typename");
    auto inout       = array("inout", "scalar_type *");
    auto rw          = variable("rw", "bool");
    auto thread      = variable("thread", "int");
    auto twiddles    = array("twiddles", "const scalar_type *");
    auto stride_in   = variable("stride_in", "int");
    auto stride_out  = variable("stride_out", "int");
    auto offset_in   = variable("offset_in", "int");
    auto offset_out  = variable("offset_out", "int");
    auto offset_lds  = variable("offset_lds", "int");

    fft->templates.push_back(scalar_type->argument());
    fft->arguments.push_back(inout->argument());
    fft->arguments.push_back(rw->argument());
    fft->arguments.push_back(thread->argument());
    fft->arguments.push_back(twiddles->argument());
    fft->arguments.push_back(stride_in->argument());
    fft->arguments.push_back(stride_out->argument());
    fft->arguments.push_back(offset_in->argument());
    fft->arguments.push_back(offset_out->argument());
    fft->arguments.push_back(offset_lds->argument());

    //
    // variables definitions
    //

    // XXX lds size
    auto lds       = array("lds", "__shared__ scalar_type", literal(1024));
    auto registers = array("R", "scalar_type", literal(factors[0] * 2));

    fft->body.push_back(lds->declaration());
    fft->body.push_back(registers->declaration());

    auto W = scalar("W", "scalar_type");
    auto t = scalar("t", "scalar_type");

    fft->body.push_back(W->declaration());
    fft->body.push_back(t->declaration());

    fft->body.push_back(line_break());

    // shortcuts
    auto Z = *inout;
    auto X = *lds;
    auto R = *registers;
    auto T = *twiddles;

    //
    // pass 0: flip/flop load from global; butterfly right away; write to lds
    //

    auto width  = factors[0];
    auto height = factors[1];
    for(int h = 0; h < height; ++h)
    {
        // load
        for(int w = 0; w < width; ++w)
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
            fft->body.push_back(assign(R[width * (h % 2) + w], Z[idx]));
        }

        // butterly

        auto fwd = call("FwdRad" + std::to_string(width) + "B1");
        for(int w = 0; w < width; ++w)
            fwd->arguments.push_back(R[(h % 2) * width + w]->address());
        fft->body.push_back(fwd);

        // write to lds
        for(int w = 0; w < width; ++w)
        {
            // clang-format off
            auto base = group(add({multiply({literal(height), thread}), literal(h)}));
            auto idx = add({
                offset_lds,
                multiply({group(base), literal(width)}),
                literal(w)});
            // clang-format on
            fft->body.push_back(assign(X[idx], R[(h % 2) * width + w]));
        }
        fft->body.push_back(line_break());
    }

    //
    // subsequent passes: flip/flop load from lds; butterfly; write to lds (or global)
    //
    auto unique = unique_factors(factors);

    for(int pass = 1; pass < factors.size(); ++pass)
    {
        width        = factors[pass];
        height       = product(unique) / width;
        auto nheight = product(factors, pass);

        for(int h = 0; h < height; ++h)
        {

            // load
            for(int w = 0; w < width; ++w)
            {
                // clang-format off
                auto idx =
                  add({
                      offset_lds,
                      multiply({literal(height), thread}),
                      literal((length / width) * w + h)});
                // clang-format on
                fft->body.push_back(assign(R[(h % 2) * width + w], X[idx]));
            }

            // twiddle
            for(int w = 1; w < width; ++w)
            {
                // clang-format off
                auto tidx =
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
                auto ridx = (h % 2) * width + w;
                fft->body.push_back(assign(W, T[tidx]));
                fft->body.push_back(assign(
                    t->x, sub({multiply({W->x, R[ridx]->x}), multiply({W->y, R[ridx]->y})})));
                fft->body.push_back(assign(
                    t->y, add({multiply({W->y, R[ridx]->x}), multiply({W->x, R[ridx]->y})})));
                fft->body.push_back(assign(R[ridx], t));
            }

            // butterly
            auto fwd = call("FwdRad" + std::to_string(width) + "B1");
            for(int w = 0; w < width; ++w)
                fwd->arguments.push_back(R[(h % 2) * width + w]->address());
            fft->body.push_back(fwd);

            // write
            if(pass < factors.size() - 1)
            {
                // write to lds
                for(int w = 0; w < width; ++w)
                {
                    // clang-format off
                    auto base = group(add({multiply({literal(height), thread}), literal(h)}));
                    auto idx =
                      add({
                          offset_lds,
                          group(add({
                                multiply({group(divide({base, literal(nheight)})), literal(width*nheight)}),
                                mod({base, literal(nheight)}),
                                literal(w*nheight)}))});
                    // clang-format on
                    fft->body.push_back(assign(X[idx], R[(h % 2) * width + w]));
                }
            }
            else
            {
                // write to output
                height = factors[0];
                width  = product(unique) / height;

                for(int w = 0; w < width; ++w)
                {
                    // clang-format off
                    auto idx =
                      add({
                          offset_out,
                          multiply({
                              group(add({
                                    multiply({literal(height), thread}),
                                    literal((length / width) * w + h)})),
                              stride_out
                            })});
                    // clang-format on
                    fft->body.push_back(assign(Z[idx], R[width * (h % 2) + w]));
                }
            }
            fft->body.push_back(line_break());
        }
    }

    return fft;
}

int main(int argc, char* argv[])
{
    std::vector<int> factors = {8, 7};
    auto             kernel  = make_device_fft(factors);
    std::cout << kernel->render() << std::endl;
}
