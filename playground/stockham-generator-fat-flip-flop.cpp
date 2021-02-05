//
// Simple AST based code generator for Stockham kernels.
//

#include <memory>
#include <vector>

#include "generator.hpp"

using namespace gen;

//
// Stockham FFT.
//

struct LaunchParams
{
    int thread_per_batch;
    int batch_per_block;
};

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

LaunchParams get_launch_params(std::vector<int> const& factors, int threads_per_block = 64)
{
    LaunchParams params;
    auto         length             = product(factors);
    auto         outputs_per_thread = factors[0] * factors[1];
    params.thread_per_batch         = length / outputs_per_thread;
    params.batch_per_block          = threads_per_block / params.thread_per_batch;
    return params;
}

std::shared_ptr<Function> make_device_fft(std::vector<int> factors, int working_sets = -1)
{
    //
    // function and argument definitions
    //
    auto length      = product(factors);
    auto fft         = function("forward_length" + std::to_string(length) + "_device");
    auto scalar_type = variable("scalar_type", "typename");
    auto inout       = array("inout", "scalar_type *");
    auto lds         = array("lds", "scalar_type *");
    auto thread      = variable("thread", "int");
    auto twiddles    = array("twiddles", "const scalar_type *");
    auto stride_in   = variable("stride_in", "int");
    auto stride_out  = variable("stride_out", "int");
    auto offset_in   = variable("offset_in", "int");
    auto offset_out  = variable("offset_out", "int");
    auto offset_lds  = variable("offset_lds", "int");

    fft->type_qualifier = DEVICE;
    fft->templates.push_back(scalar_type->argument());
    fft->arguments.push_back(inout->argument());
    fft->arguments.push_back(lds->argument());
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

    if(working_sets < 0)
        working_sets = factors[1];

    int  nregisters = factors[0] * working_sets;
    auto registers  = array("R", "scalar_type", literal(nregisters));
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
    // pass 0: pipelined load from global + butterfly right away + write to lds
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
            fft->body.push_back(assign(R[(width * h + w) % nregisters], Z[idx]));
        }

        // butterly

        auto fwd = call("FwdRad" + std::to_string(width) + "B1");
        for(int w = 0; w < width; ++w)
            fwd->arguments.push_back(R[(h * width + w) % nregisters]->address());
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
            fft->body.push_back(assign(X[idx], R[(h * width + w) % nregisters]));
        }
        fft->body.push_back(line_break());
    }

    //
    // subsequent passes: pipelined load from lds + butterfly + write
    //
    auto unique = unique_factors(factors);

    for(int pass = 1; pass < factors.size(); ++pass)
    {
        width        = factors[pass];
        height       = factors[0] * factors[1] / width;
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
                fft->body.push_back(assign(R[(h * width + w) % nregisters], X[idx]));
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
                auto ridx = (h * width + w) % nregisters;
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
                fwd->arguments.push_back(R[(h * width + w) % nregisters]->address());
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
                    fft->body.push_back(assign(X[idx], R[(h * width + w) % nregisters]));
                }
            }
            else
            {
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
                    fft->body.push_back(assign(Z[idx], R[(h * width + w) % nregisters]));
                }
            }
            fft->body.push_back(line_break());
        }
    }

    return fft;
}

std::shared_ptr<Function> make_global_fft(std::vector<int> factors)
{
    auto length = product(factors);
    auto fft    = function("forward_length" + std::to_string(length));

    auto scalar_type = variable("scalar_type", "typename");
    auto inout       = array("inout", "scalar_type *");
    auto twiddles    = array("twiddles", "const scalar_type *");
    auto stride_in   = variable("stride_in", "int");
    auto stride_out  = variable("stride_out", "int");
    auto nbatch      = variable("nbatch", "int");

    fft->type_qualifier = GLOBAL;
    fft->templates.push_back(scalar_type->argument());
    fft->arguments.push_back(inout->argument());
    fft->arguments.push_back(nbatch->argument());
    fft->arguments.push_back(twiddles->argument());
    fft->arguments.push_back(stride_in->argument());
    fft->arguments.push_back(stride_out->argument());

    auto params = get_launch_params(factors);

    auto lds = array("lds", "__shared__ scalar_type", literal(length * params.batch_per_block));
    fft->body.push_back(lds->declaration());
    fft->body.push_back(line_break());

    auto thread     = variable("thread", "int");
    auto block_id   = scalar("blockIdx.x");
    auto thread_id  = scalar("threadIdx.x");
    auto offset_in  = variable("offset_in", "int");
    auto offset_out = variable("offset_out", "int");
    auto offset_lds = variable("offset_lds", "int");
    auto batch      = variable("batch", "int");

    fft->body.push_back(thread->declaration());
    fft->body.push_back(batch->declaration());
    fft->body.push_back(offset_in->declaration());
    fft->body.push_back(offset_out->declaration());
    fft->body.push_back(offset_lds->declaration());

    fft->body.push_back(assign(thread, mod({thread_id, literal(params.thread_per_batch)})));
    fft->body.push_back(assign(batch,
                               add({multiply({literal(params.batch_per_block), block_id}),
                                    divide({thread_id, literal(params.thread_per_batch)})})));

    fft->body.push_back(assign(offset_in, multiply({literal(length), batch})));
    fft->body.push_back(assign(offset_out, multiply({literal(length), batch})));
    fft->body.push_back(
        assign(offset_lds,
               multiply({literal(length), group(mod({batch, literal(params.batch_per_block)}))})));

    auto device = call("forward_length" + std::to_string(length) + "_device");
    device->templates.push_back(scalar_type);
    device->arguments.push_back(inout);
    device->arguments.push_back(lds);
    device->arguments.push_back(thread);
    device->arguments.push_back(twiddles);
    device->arguments.push_back(stride_in);
    device->arguments.push_back(stride_out);
    device->arguments.push_back(offset_in);
    device->arguments.push_back(offset_out);
    device->arguments.push_back(offset_lds);
    fft->body.push_back(device);

    return fft;
}

std::shared_ptr<Function> make_host_fft(std::vector<int> factors)
{
    auto length = product(factors);
    auto fft    = function("forward_length" + std::to_string(length) + "_launch");

    auto scalar_type = variable("scalar_type", "typename");
    auto inout       = array("inout", "scalar_type *");
    auto twiddles    = array("twiddles", "const scalar_type *");
    auto stride_in   = variable("stride_in", "int");
    auto stride_out  = variable("stride_out", "int");
    auto nbatch      = variable("nbatch", "int");

    fft->templates.push_back(scalar_type->argument());
    fft->arguments.push_back(inout->argument());
    fft->arguments.push_back(nbatch->argument());
    fft->arguments.push_back(twiddles->argument());
    fft->arguments.push_back(stride_in->argument());
    fft->arguments.push_back(stride_out->argument());

    auto global = call("forward_length" + std::to_string(length) + "");
    global->templates.push_back(scalar_type);
    global->arguments.push_back(inout);
    global->arguments.push_back(nbatch);
    global->arguments.push_back(twiddles);
    global->arguments.push_back(stride_in);
    global->arguments.push_back(stride_out);

    auto params = get_launch_params(factors);

    if(params.thread_per_batch <= 32)
    {
        auto nblocks = variable("nblocks", "int");
        fft->body.push_back(nblocks->declaration());
        fft->body.push_back(
            assign(nblocks,
                   divide({group(add({nbatch, literal(params.batch_per_block - 1)})),
                           literal(params.batch_per_block)})));
        global->kernel_arguments.push_back(nblocks);
        global->kernel_arguments.push_back(
            literal(params.thread_per_batch * params.batch_per_block));
    }
    else
    {
        global->kernel_arguments.push_back(nbatch);
        global->kernel_arguments.push_back(literal(params.thread_per_batch));
    }
    fft->body.push_back(global);

    return fft;
}

int main(int argc, char* argv[])
{
    std::vector<int> factors = {7, 6, 2, 2, 2};

    auto device = make_device_fft(factors, 1);
    auto global = make_global_fft(factors);
    auto host   = make_host_fft(factors);

    std::cout << device->render() << std::endl;
    std::cout << global->render() << std::endl;
    std::cout << host->render() << std::endl;
}
