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
    
    for ibatch in range(batch):
        Zlength = np.append(length[0:-1], length[-1] // 2)
        z = np.zeros(shape=Zlength, dtype=complex)
        for idx in (list(itertools.product(*[range(l) for l in Zlength]))):
            ridx0 = list(idx)
            ridx0[-1] *= 2
            ridx1 = list(idx)
            ridx1[-1] *= 2
            ridx1[-1] += 1
            # Read op here.
            z[idx] = complex(x[ibatch][tuple(ridx0)], x[ibatch][tuple(ridx1)])
        Z = np.fft.fft(z)
        for idx in (list(itertools.product(*[range(l) for l in length[0:-1]]))):
            X[ibatch][idx] = postkernel(Z[idx])

    for dim in range(len(Zlength) - 1):
        X = np.fft.fft(X, axis=dim+1)

    # Write op here.
    
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
