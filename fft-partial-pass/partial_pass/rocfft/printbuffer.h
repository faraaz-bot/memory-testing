// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef PRINTBUFFER_H
#define PRINTBUFFER_H

#include "increment.h"
#include <algorithm>
#include <vector>

// Output a formatted general-dimensional array with given length and stride in batches
// separated by dist.
template <typename Toutput, typename T1, typename T2, typename Tsize, typename Tstream>
inline void printbuffer(const Toutput *output,
                        const std::vector<T1> &length,
                        const std::vector<T2> &stride,
                        const Tsize nbatch,
                        const Tsize dist,
                        const size_t offset,
                        Tstream &stream)
{
    auto i_base = 0;
    for (unsigned int b = 0; b < nbatch; b++, i_base += dist)
    {
        std::vector<size_t> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            const int i = std::inner_product(index.begin(), index.end(), stride.begin(), i_base + offset);
            stream << output[i] << " ";
            for (int li = index.size(); li-- > 0;)
            {
                if (index[li] == (length[li] - 1))
                {
                    //stream << "\n";
                }
                else
                {
                    break;
                }
            }
        } while (increment_rowmajor(index, length));
        //stream << std::endl;
    }
}

template <typename Telem, typename Tint1, typename Tint2, typename Tsize, typename Tstream = std::ostream>
static void print_buffer(const std::vector<Telem> &buf,
                         const std::vector<Tint1> &length,
                         const std::vector<Tint2> &stride,
                         const Tsize nbatch,
                         const Tsize dist,
                         Tstream &stream = std::cout)
{

    printbuffer(reinterpret_cast<const Telem *>(buf.data()),
                length,
                stride,
                nbatch,
                dist,
                0,
                stream);
};

#endif
