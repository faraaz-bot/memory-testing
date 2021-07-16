
#include "generator.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>

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

Function make_device_fft(std::vector<int> factors)
{
    int length                = product(factors);
    int threads_per_transform = length / factors[0];

    auto kdevice = Function("forward_length" + std::to_string(length));

    auto R          = Variable("R");
    auto thread     = Variable("thread");
    auto thread_id  = Variable("threadIdx.x");
    auto lds        = Variable("lds");
    auto offset_lds = Variable("offset_lds");
    auto W          = Variable("W");
    auto t          = Variable("t");
    auto twiddles   = Variable("twiddles");

    kdevice.body += R.declaration();
    kdevice.body += thread.declaration();
    kdevice.body += Assign(thread, thread_id % threads_per_transform);

    for(int pass = 0; pass < factors.size(); ++pass)
    {
        auto width   = factors[pass];
        auto nheight = product(factors, pass);
        auto height  = double(length) / width / threads_per_transform;
        auto iheight = floor(height);

        for(auto width : factors)
        {
            // load lds
            for(int h = 0; h < iheight; ++h)
            {
                for(int w = 0; w < width; ++w)
                {
                    auto tid = thread + h * threads_per_transform;
                    auto idx = offset_lds + tid + w * (length / width);
                    kdevice.body += Assign(R[h * width + w], lds[idx]);
                }
            }

            // apply twiddle
            for(int h = 0; h < iheight; ++h)
            {
                for(int w = 1; w < width; ++w)
                {
                    auto tid  = thread + h * threads_per_transform;
                    auto tidx = offset_lds + tid + w * (length / width);
                    auto ridx = h * width + w;
                    kdevice.body += Assign(W, twiddles[tidx]);
                    kdevice.body += Assign(t.x, W.x * R[ridx].x - W.y * R[ridx].y);
                    kdevice.body += Assign(t.y, W.y * R[ridx].x + W.x * R[ridx].y);
                    kdevice.body += Assign(R[ridx], t);
                }
            }

            // butterfly
            for(int h = 0; h < iheight; ++h)
            {
                auto args = ArgumentList();
                for(int w = 0; w < width; ++w)
                    args.arguments.push_back(R[w].address());
                kdevice.body += Call("FwdRad" + std::to_string(width), args);
            }

            // store lds
            for(int h = 0; h < iheight; ++h)
            {
                for(int w = 0; w < width; ++w)
                {
                    auto tid = thread + h * threads_per_transform;
//                    auto idx = offset_lds + B(B(tid / cumheight) * (width * cumheight) + tid % cumheight + w * cumheight) * lstride;
                    auto idx = offset_lds + w; // XXX
                    kdevice.body += Assign(lds[idx], R[h * width + w]);

                }
            }
        }
    }

    return kdevice;
}

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

    auto device = make_device_fft(factors);

    format_and_write("stockham_generated_kernel.h", device.render());
}
