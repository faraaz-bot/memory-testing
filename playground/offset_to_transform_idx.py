# yield all of the offsets reachable from a specified length + stride
def get_offsets(length, stride):
    if len(length) == 0:
        yield 0
    else:
        for offset in get_offsets(length[1:], stride[1:]):
            for i in range(length[0]):
                yield i * stride[0] + offset


class IoDim:
    def __init__(self, length, stride):
        self.length = length
        self.stride = stride
        # True if this is the FFT dimension
        self.fft_dim = False
    def __lt__(self, other):
        return self.stride < other.stride
    def __repr__(self):
        return "len={} stride={} fft={}".format(self.length, self.stride, self.fft_dim)

def iodims_collapsible(iodim1, iodim2):
    if iodim1.fft_dim or iodim2.fft_dim:
        return False
    return iodim1.length * iodim1.stride == iodim2.stride
    
if __name__ == '__main__':
    lengths = [4,4,4]
    strides = [1,4,32]

    # gather iodims, sort them from fastest to slowest.
    iodims = []
    for length, stride in zip(lengths, strides):
        iodims.append(IoDim(length,stride))
    # pick a dimension to be the one we're transforming
    iodims[0].fft_dim = True
    iodims.sort()

    # identify the FFT dim
    fft_dims = list(filter(lambda e : e[1].fft_dim, enumerate(iodims)))
    if len(fft_dims) != 1:
        raise RuntimeError("need exactly 1 fft dim")
    fft_dim = fft_dims[0][0]

    # discard any faster dims than the FFT dim
    del iodims[0:fft_dim]

    # collapse any contiguous higher dims if possible
    while len(iodims) >= 2 and iodims_collapsible(iodims[-2], iodims[-1]):
        iodims[-2].length *= iodims[-1].length
        iodims.pop()

    print("{} divs needed to compute transform index".format(len(iodims)))

    print(iodims)
    
    for offset in get_offsets(lengths, strides):
        # for each reachable offset, work backwards to get the
        # indexes used to reach that offset
        cur_offset = offset
        cur_index = 0
        for iodim in reversed(iodims):
            cur_index = cur_offset // iodim.stride
            cur_offset = cur_offset % iodim.stride
        print("offset {} transform_index={}".format(offset, cur_index))
    
