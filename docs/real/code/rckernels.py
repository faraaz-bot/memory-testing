import math
import cmath
import numpy as np

def postkernel(Z):
    Nhalf = len(Z)
    N = 2 * Nhalf
    Ncomplex = Nhalf + 1
    X = np.empty(Ncomplex, dtype=complex)
    
    # X[0] = complex(Z[0].real + Z[0].imag, 0)
    # I = complex(0, 1)
    # for r in range(1, Nhalf):
    #     omegaNr = cmath.exp(-2.0 * math.pi * I * r / N);
    #     X[r] = Z[r] * 0.5 * (1 - I * omegaNr) + Z[Nhalf - r].conjugate() * 0.5 * (1 + I * omegaNr)
    # X[Nhalf] = complex(Z[0].real - Z[0].imag, 0)

    X[0] = complex(Z[0].real + Z[0].imag, 0)
    I = complex(0, 1)
    if Nhalf % 2 == 0:
        p = Nhalf // 2
        print(p)
        omegaNp = -I
        print(omegaNp.imag)
        X[p] = Z[p].conjugate()
    for p in range(1, Nhalf // 2):
        print(p)
        q = Nhalf - p
        omegaNp = cmath.exp(-2.0 * math.pi * I * p / N);
        X[p] = Z[p] * 0.5 * (1 - I * omegaNp) + Z[q].conjugate() * 0.5 * (1 + I * omegaNp)
        omegaNq = -omegaNp.conjugate()
        X[q] = Z[q] * 0.5 * (1 - I * omegaNq) + Z[p].conjugate() * 0.5 * (1 + I * omegaNq)
    X[Nhalf] = complex(Z[0].real - Z[0].imag, 0)

    
    return X
                                
def prekernel(X):
    Nhalf = len(X) - 1
    N = 2 * Nhalf
    Z = np.empty(Nhalf, dtype=complex)
    I = complex(0, 1)
    
    # for r in range(0, Nhalf):
    #     omegaNr = cmath.exp(2.0 * math.pi * I * r / N);
    #     Z[r] = X[r] * (1 + I * omegaNr) + X[Nhalf - r].conjugate() * (1 - I * omegaNr)
    
    # Xp = X[0]
    # Xq = X[Nhalf]
    # Z[0] = complex(Xp.real - Xp.imag + Xq.real + Xq.imag,
    #                Xp.real + Xp.imag - Xq.real + Xq.imag)
    # for p in range(1, (Nhalf) // 2 + 1):
    #     print(p)
    #     q = Nhalf - p
    #     omegaNp = cmath.exp(2.0 * math.pi * I * p / N);
    #     omegaNq = -omegaNp.conjugate()
    #     Z[p] = X[p] * (1 + I * omegaNp) + X[q].conjugate() * (1 - I * omegaNp)
    #     Z[q] = X[q] * (1 + I * omegaNq) + X[p].conjugate() * (1 - I * omegaNq)

    # The twiddle table only needs to be 1/4 the length of the transform.
    twids = np.empty(Nhalf // 2, dtype=complex)
    for k in range(len(twids)):
        twids[k] = cmath.exp(2.0 * math.pi * I * k / N)
    
    Xp = X[0]
    Xq = X[Nhalf]
    Z[0] = complex(Xp.real - Xp.imag + Xq.real + Xq.imag,
                   Xp.real + Xp.imag - Xq.real + Xq.imag)
    if Nhalf % 2 == 0:
        Z[Nhalf // 2] =  X[Nhalf // 2].conjugate() * 2
    for p in range(1, (Nhalf + 1) // 2):
        q = Nhalf - p
        #omegaNp = cmath.exp(2.0 * math.pi * I * p / N)
        omegaNp = twids[p] if p < Nhalf // 2  else -twids[p].conjugate
        if not omegaNp == cmath.exp(2.0 * math.pi * I * p / N):
            print("error in twiddle computation at index", p) 
        omegaNq = -omegaNp.conjugate()
        Z[p] = X[p] * (1 + I * omegaNp) + X[q].conjugate() * (1 - I * omegaNp)
        Z[q] = X[q] * (1 + I * omegaNq) + X[p].conjugate() * (1 - I * omegaNq)
        print((omegaNq).imag)
        
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
