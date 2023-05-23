import math
import cmath
import numpy as np
import copy
import itertools

# Perform a general-dimensional batched real-to-complex even-length FFT.
def rcfft_even(x, length, batch, readop=None, writeop=None):
    if len(length) == 0:
        raise ValueError("No lengths were provided")
    if length[-1] %2 != 0:
        raise ValueError("Last dimension is not even")
    hlength = copy.deepcopy(length)
    hlength[-1] = hlength[-1] // 2 + 1 
    X = np.zeros(shape=np.append(batch, hlength), dtype=complex)

    # The real-to-complex dimension:
    for ibatch in range(batch):
        Zlength = np.append(length[0:-1], length[-1] // 2)
        z = np.zeros(shape=Zlength, dtype=complex)
        for idx in (list(itertools.product(*[range(l) for l in Zlength]))):
            lidx = list(idx)
            ridx0 = lidx[0:-1]
            ridx0.append(lidx[-1] * 2)
            ridx1 = lidx[0:-1]
            ridx1.append(lidx[-1] * 2 + 1)
            # Read op here.
            if readop == None:
                z[idx] = complex(x[ibatch][tuple(ridx0)], x[ibatch][tuple(ridx1)])
                #z[idx] = x[ibatch][tuple(ridx0)] + ij * x[ibatch][tuple(ridx1)]
            else:
                # TODO: instead of complex addition, just use complex(a,b) to ensure that the
                # read-op is real-to-real?
                z[idx] = readop(x[ibatch][tuple(ridx0)],ibatch,ridx0) \
                    + 1j * readop(x[ibatch][tuple(ridx1)],ibatch,ridx1) 
        Z = np.fft.fft(z)
        for idx in (list(itertools.product(*[range(l) for l in length[0:-1]]))):
            X[ibatch][idx] = postkernel(Z[idx])

    # Complex-to-complex transform on all of the non-batch dimensions:
    for dim in range(len(Zlength) - 1):
        X = np.fft.fft(X, axis = dim + 1)

    # Write op here.
    if writeop != None:
        for ibatch in range(batch):
            for idx in (list(itertools.product(*[range(l) for l in hlength]))):
                X[ibatch][idx] = writeop(X[idx], ibatch, idx)
    
    return X

# Perform a general-dimensional batched real-to-complex even-length FFT.
def crfft_even(X, length, batch, readop=None, writeop=None):
    if len(length) == 0:
        raise ValueError("No lengths were provided")
    if length[-1] %2 != 0:
        raise ValueError("Last dimension is not even")
    x = np.zeros(shape=np.append(batch, length), dtype=float)

    if readop != None:
        for ibatch in range(batch):
            for idx in (list(itertools.product(*[range(l) for l in length]))):
                x[ibatch][idx] = writeop(x[idx], ibatch, idx)
    
    # Complex-to-complex transform on all of the non-batch dimensions:
    for dim in range(len(length) - 1):
        X = np.fft.ifft(X, axis = dim + 1) * length[dim]

    for ibatch in range(batch):
        Zlength = np.append(length[0:-1], length[-1] // 2)
        Z = np.zeros(shape=Zlength, dtype=complex)
        for idx in (list(itertools.product(*[range(l) for l in length[0:-1]]))):
            Z[idx] = prekernel(X[ibatch][idx])
            Z[idx] = np.fft.ifft(Z[idx]) * len(Z[idx])
        
        for idx in (list(itertools.product(*[range(l) for l in Zlength]))):
            lidx = list(idx)
            ridx0 = lidx[0:-1]
            ridx0.append(lidx[-1] * 2)
            ridx1 = lidx[0:-1]
            ridx1.append(lidx[-1] * 2 + 1)
            if writeop == None:
                x[ibatch][tuple(ridx0)] = Z[idx].real
                x[ibatch][tuple(ridx1)] = Z[idx].imag
            else:
                x[ibatch][tuple(ridx0)] = writeop(Z[idx].real, ibatch, ridx0)
                x[ibatch][tuple(ridx1)] = writeop(Z[idx].imag, ibatch, ridx1)
        
    return x

# Perform a general-dimensional batched real-to-complex FFT via complex embedding.
def rcfft_embed(x, length, batch, readop=None, writeop=None):
    if len(length) == 0:
        raise ValueError("No lengths were provided")
    hlength = copy.deepcopy(length)
    hlength[-1] = hlength[-1] // 2 + 1 
    X = np.zeros(shape=np.append(batch, hlength), dtype=complex)

    
    # The real-to-complex dimension:
    for ibatch in range(batch):
        for idx in (list(itertools.product(*[range(l) for l in length[:-1]]))):
            Z = np.zeros(length[-1], dtype=complex)
            for idx0 in range(length[-1]):
                # TODO: read-op
                if readop == None:
                    Z[idx0] = x[ibatch][idx][idx0]
                else:
                    idxx = list(idx)
                    idxx.append(idx0)
                    Z[idx0] = readop(x[ibatch][idx][idx0],ibatch,idxx)
            Z = np.fft.fft(Z)
            for idx0 in range(hlength[-1]):
                X[ibatch][idx][idx0] = Z[idx0]

    # TODO: what if we apply the non-Hermitian symmetric readop here?  For shift at least.
    # Complex-to-complex transform on all of the non-batch dimensions:
    for dim in range(len(hlength) - 1):
        X = np.fft.fft(X, axis = dim + 1)
    
    # Write op here.
    if writeop != None:
        for ibatch in range(batch):
            for idx in (list(itertools.product(*[range(l) for l in hlength]))):
                X[ibatch][idx] = writeop(X[idx], ibatch, idx)
    
    return X
    
# Perform a general-dimensional batched real-to-complex even-length FFT.
def rcfft_pair(x, length, batch, readop=None, writeop=None):
    if len(length) == 0:
        raise ValueError("No lengths were provided")
    otherlength = batch * np.prod(length[:-1])
    if otherlength %2 != 0:
        raise ValueError("not an even number somewhere")
    
    hlength = copy.deepcopy(length)
    hlength[-1] = hlength[-1] // 2 + 1 
    X = np.zeros(shape=np.append(batch, hlength), dtype=complex)
    
    x0 = np.reshape(x, [otherlength, length[-1]])
    X = np.reshape(X, [otherlength, hlength[-1]])
    for idx0 in range(otherlength // 2):
        
        z = np.empty([length[-1]], dtype=complex)
        for idx1 in range(length[-1]):
            pp0 = np.unravel_index(np.ravel_multi_index((idx0*2, idx1), x0.shape), x.shape)
            pp1 = np.unravel_index(np.ravel_multi_index((idx0*2 + 1, idx1), x0.shape), x.shape)
            batch0 = pp0[0]
            idx00 = pp0[1:]
            batch1 = pp1[0]
            idx11 = pp1[1:]
            
            if readop == None:
                z[idx1] = complex(x0[idx0 * 2][idx1], x0[idx0 * 2 + 1][idx1])
            else:
                z[idx1] = readop(x0[idx0 * 2][idx1], batch0, idx00) \
                    + 1j * readop(x0[idx0 * 2 + 1][idx1], batch1, idx11)
        Z = np.fft.fft(z)
        X[idx0 * 2], X[idx0 * 2 + 1] = unpackbatch(Z)
        
    X = np.reshape(X, np.append(batch, hlength))

    for dim in range(len(hlength) - 1):
        X = np.fft.fft(X, axis = dim + 1)
           
    if writeop != None:
        for ibatch in range(batch):
            for idx in (list(itertools.product(*[range(l) for l in hlength]))):
                X[ibatch][idx] = writeop(X[idx], ibatch, idx) 
    return X

    
def rfft(x, length, batch):
    # x: real input data
    # length: array-like int
    # batch: int
    if len(length) == 0:
        X = np.array(length, dtype=complex)
        X = x
    if length[-1] % 2 == 0:
        print("even!")
    else:
        print("odd!")

    

def postkernel(Z):
    # Real-to-complex post kernel.
    Nhalf = len(Z)
    N = 2 * Nhalf
    Ncomplex = Nhalf + 1
    
    X = np.empty(Ncomplex, dtype=complex)

    I = complex(0, 1)
    
    version = "half"
    
    if version == "simple":
        X[0] = complex(Z[0].real + Z[0].imag, 0)
        for p in range(1, Nhalf):
            omegaNp = cmath.exp(-2.0 * math.pi * I * p / N);
            X[p] = Z[p] * 0.5 * (1 - I * omegaNp) \
                + Z[Nhalf - p].conjugate() * 0.5 * (1 + I * omegaNp)
        X[Nhalf] = complex(Z[0].real - Z[0].imag, 0)
        return X
    elif version == "half":
        twid = np.empty((Nhalf + 1) // 2, dtype=complex)
        for p in range(len(twid)):
            twid[p] = cmath.exp(-2.0 * math.pi * I * p / N)
        
        X[0] = complex(Z[0].real + Z[0].imag, 0)
        for p in range(1, (Nhalf + 1) // 2):
            q = Nhalf - p
            omegaNp = twid[p]
            if omegaNp != cmath.exp(-2.0 * math.pi * I * p / N):
                print("failure in twiddle computation")
            #omegaNp = cmath.exp(-2.0 * math.pi * I * p / N);
            #omegaNq = cmath.exp(-2.0 * math.pi * I * q / N);
            omegaNq = -omegaNp.conjugate()
            X[p] = Z[p] * 0.5 * (1 - I * omegaNp) \
                + Z[q].conjugate() * 0.5 * (1 + I * omegaNp)
            X[q] = Z[q] * 0.5 * (1 - I * omegaNq) \
                + Z[p].conjugate() * 0.5 * (1 + I * omegaNq)
        if Nhalf % 2 == 0:
            p = Nhalf // 2
            omegaNp = -I
            X[p] = Z[p].conjugate()
        X[Nhalf] = complex(Z[0].real - Z[0].imag, 0)
        return X
        

def prekernel(X):
    # Complex-to-real pre kernel
    Nhalf = len(X) - 1
    N = 2 * Nhalf
    Z = np.empty(Nhalf, dtype=complex)
    I = complex(0, 1)

    version = "half"
    
    if version == "simple":
        # Simplest version.
        for p in range(0, Nhalf):
            omegaNp = cmath.exp(2.0 * math.pi * I * p / N);
            Z[p] = X[p] * (1 + I * omegaNp) \
                + X[Nhalf - p].conjugate() * (1 - I * omegaNp)
        return Z
    elif version == "half":
        # p and Nhalf - p are both used in the same computation, so we
        # can save a twiddle and a r/w.
       
        twid = np.empty((Nhalf + 1) // 2, dtype=complex)
        for p in range(len(twid)):
            twid[p] = cmath.exp(2.0 * math.pi * I * p / N)
            
        Xp = X[0]
        Xq = X[Nhalf]
        Z[0] = complex(Xp.real - Xp.imag + Xq.real + Xq.imag,
                       Xp.real + Xp.imag - Xq.real + Xq.imag)
        
        if Nhalf % 2 == 0:
            Z[Nhalf // 2] =  X[Nhalf // 2].conjugate() * 2
            
        for p in range(1, (Nhalf + 1) // 2):
            omegaNp = twid[p]
            if omegaNp != cmath.exp(2.0 * math.pi * I * p / N):
                print("failure in twiddle computation")
            q = Nhalf - p
            omegaNq = -omegaNp.conjugate()
            Z[p] = X[p] * (1 + I * omegaNp) \
                + X[q].conjugate() * (1 - I * omegaNp)
            Z[q] = X[q] * (1 + I * omegaNq) \
                + X[p].conjugate() * (1 - I * omegaNq)
        return Z        

def unpackbatch(Z):
    dim = len(Z.shape)
    N = Z.shape[dim - 1]
    Np = N //2 + 1
    hermshape = list(Z.shape)
    hermshape[dim - 1] = Np
    X = np.empty(hermshape, dtype=complex)
    Y = np.empty(hermshape, dtype=complex)
    I = complex(0, 1)
    idxshape = hermshape[0:dim-2] if dim > 1 else [0]
    X[0] = complex(Z[0].real, 0)
    Y[0] = complex(Z[0].imag, 0)
    for r in range(1, Np):
        X[r] = 0.5 * (Z[r] + Z[N - r].conjugate())
        Y[r] = -I * 0.5 * (Z[r] - Z[N - r].conjugate())
    return X, Y

def repackbatch(X, Y, N):
    Z = np.empty(N, dtype=complex)
    Np = len(X)
    Nhalf = Np - 1
    even = 2*(Np - 1) == N
    I = complex(0, 1)
    Z[0] = complex(X[0].real, Y[0].real)
    # NB: It's also correct to always have stop Np.
    stop = Nhalf if even else Np
    for r in range(1, stop):
        Z[r] = X[r] + I * Y[r]
        Z[N - r] = X[r].conjugate() + I * Y[r].conjugate()
    if even:
        Z[stop] = X[stop] + I * Y[stop]
    return Z

def unpackND(Xplanar, hermshape):
    dim = len(Xplanar.shape)
    N = Xplanar.shape[dim - 1]
    Np = N // 2 + 1
    nbatch = 1
    for d in Xplanar.shape[0:dim - 1]:
        nbatch *= d
    Xplanar.reshape([d, N])
    X = np.empty([d * 2, Np] , dtype=complex)
    I = complex(0, 1)
    for idx in range(nbatch):
        idxe = 2 * idx
        idxo = 2 * idx + 1
        X[idxe, 0] = complex(Xplanar[idx, 0].real, 0)
        X[idxo, 0] = complex(Xplanar[idx, 0].imag, 0)
        for r in range(1, Np):
            X[idxe, r] = 0.5 * (Xplanar[idx, r] + Xplanar[idx, N - r].conjugate())
            X[idxo, r] = -I * 0.5 * (Xplanar[idx, r] - Xplanar[idx, N - r].conjugate())
    X.reshape(hermshape)
    return X

def repackND(X, lengths):
    # Given a complex-Hermitian array X with at least one even higher
    # dimension, untangle and get the real data of length lengths,
    # which we will then do a c2c-inv on.
    # X has dimensions [Nx, Ny // 2 + 1]
    # Xplanar has dimensions [ Nx // 2, Ny]
    dim = len(lengths)
    N = lengths[dim - 1] # Last dimension
    Np = N // 2 + 1
    nbatch = 1
    for d in lengths[0:dim - 1]:
        nbatch *= d
    nbatch = nbatch // 2
    X.reshape([d, Np])
    Xplanar = np.empty([d // 2, N] , dtype=complex)

    even = 2*(Np - 1) == N
    stop = N // 2 if even else Np
    
    I = complex(0, 1)
    for idx in range(nbatch):
        idxe = 2 * idx
        idxo = 2 * idx + 1
        Xplanar[idx][0] = complex(X[idxe][0].real, X[idxo][0].real)
        for r in range(1, stop):
            Xplanar[idx][r] = X[idxe][r] + I * X[idxo][r]
            Xplanar[idx][N - r] = X[idxe][r].conjugate() + I * X[idxo][r].conjugate()
        if even:
            Xplanar[idx][stop] = X[idxe][stop] + I * X[idxo][stop]
    return Xplanar

def symmetrize_1d(hdata, nx):
    sdata = np.empty([nx // 2 + 1], dtype=complex)
    for i in range(nx // 2 + 1):
        sdata[i] = hdata[i]
        
    xvals = [0]
    if nx % 2 == 0:
        xvals.append(nx // 2)
        
    for xval in xvals:
        sdata[xval] = sdata[xval].real
        
    return sdata

def symmetrize_2d(hdata, nx, ny):
    sdata = np.empty([nx, ny // 2 + 1], dtype=complex)
    for i in range(nx):
        for j in range(ny // 2 + 1):
            sdata[i][j] = hdata[i][j]
            
    xvals = [0]
    if nx % 2 == 0:
        xvals.append(nx // 2)
    yvals = [0]
    if ny % 2 == 0:
        yvals.append(nx // 2)

    for yval in yvals:
        # DY/Nyquists:
        for xval in xvals:
            sdata[xval][yval] = sdata[xval][yval].real
        # x-axes:
        for i in range(1, nx // 2):
            sdata[nx - i][yval] = sdata[i][yval].conj()
            
    return sdata

def symmetrize_3d(hdata, nx, ny, nz, only_conj=False):
    sdata  = np.empty([nx, ny, nz // 2 + 1], dtype=complex)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz // 2 + 1):
                sdata[i][j][k] = hdata[i][j][k]
    
    xvals = [0]
    if nx % 2 == 0:
        xvals.append(nx // 2)
    yvals = [0]
    if ny % 2 == 0:
        yvals.append(nx // 2)
    zvals = [0]
    if nz % 2 == 0:
        zvals.append(nz // 2)
        
    for zval in zvals:
        if not only_conj:
            # DC/nyquists:
            for xval in xvals:
                for yval in yvals:
                    sdata[xval][yval][zval] = sdata[xval][yval][zval].real

        # x-axes:
        for yval in yvals:
            for i in range(1, nx // 2):
                sdata[nx - i][yval][zval] = sdata[i][yval][zval].conj()
        # y-axes:
        for xval in xvals:
            for j in range(1, ny // 2):
                sdata[xval][ny - j][zval] = sdata[xval][j][zval].conj()
        # xy-planes:
        for i in range(1, nx // 2):
            for j in range(1, ny):
                sdata[nx - i][ny - j][zval] = sdata[i][j][zval].conj()

    return sdata
