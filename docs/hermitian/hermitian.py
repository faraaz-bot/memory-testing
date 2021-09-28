#!/usr/bin/python3

import math
import cmath
import numpy as np
import random

np.set_printoptions(precision=3)

nx = 4
ny = 4
nz = 4

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

def symmetrize_3d(hdata, nx, ny, nz):
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
        
def is_symmetric_1d(hdata, nx):
    xvals = [0]
    if nx % 2 == 0:
        xvals.append(nx // 2)
    for xval in xvals:
        if not np.isclose(hdata[xval].imag, 0):
            return False
    return True

def is_symmetric_2d(hdata, nx, ny):
    xvals = [0]
    if nx % 2 == 0:
        xvals.append(nx // 2)
    yvals = [0]
    if ny % 2 == 0:
        yvals.append(nx // 2)
    for yval in yvals:
        # DY/Nyquists:
        for xval in xvals:
            if not np.isclose(hdata[xval][yval].imag, 0):
                return False
        for i in range(1, nx // 2):
            if not np.isclose(hdata[nx - i][yval], hdata[i][yval].conj()):
                return False
    return True

def is_symmetric_3d(hdata, nx, ny, nz):
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
        # DC/nyquists:
        for xval in xvals:
            for yval in yvals:
                if not np.isclose(hdata[xval][yval][zval].imag, 0):
                    return False
        # x-axes:
        for yval in yvals:
            for i in range(1, nx // 2):
                if not np.isclose(hdata[nx - i][yval][zval], \
                                  hdata[i][yval][zval].conj()):
                    return False
        # y-axes:
        for xval in xvals:
            for j in range(1, ny // 2):
                if not np.isclose(hdata[xval][ny - j][zval], \
                                  hdata[xval][j][zval].conj()):
                    return False
        # xy-planes:
        for i in range(1, nx // 2):
            for j in range(1, ny):
                if not np.isclose(hdata[nx - i][ny - j][zval], \
                                  hdata[i][j][zval].conj()):
                    return False
                
    return True
                

def r2c_1d(rdata, nx, impose_hermitian=False):
    cdata = np.empty([nx], dtype=complex)
    for i in range(nx):
        cdata[i] = rdata[i]
    cdata = np.fft.fft(cdata)
    if impose_hermitian:
        cdata = symmetrize_1d(cdata, nx)
    return cdata[0:nx // 2 + 1]

def postkernel(Z, impose_hermitian=False):
    # TODO: can we actually impose hermitian here?
    Nhalf = len(Z)
    N = 2 * Nhalf
    Ncomplex = Nhalf + 1
    hdata = np.empty(Ncomplex, dtype=complex)
    hdata[0] = complex(Z[0].real + Z[0].imag, 0)
    I = complex(0, 1)
    if Nhalf % 2 == 0:
        p = Nhalf // 2
        omegaNp = -I
        hdata[p] = Z[p].conjugate()
    for p in range(1, Nhalf // 2):
        q = Nhalf - p
        omegaNp = cmath.exp(-2.0 * math.pi * I * p / N);
        hdata[p] = Z[p] * 0.5 * (1 - I * omegaNp) \
            + Z[q].conjugate() * 0.5 * (1 + I * omegaNp)
        omegaNq = -omegaNp.conjugate()
        hdata[q] = Z[q] * 0.5 * (1 - I * omegaNq) \
            + Z[p].conjugate() * 0.5 * (1 + I * omegaNq)
    hdata[Nhalf] = complex(Z[0].real - Z[0].imag, 0)
    return hdata

def r2c_1d_even(rdata, nx, impose_hermitian=False):
    cdata = np.empty([nx // 2], dtype=complex)
    for i in range(nx // 2):
        cdata[i] = complex(rdata[2*i], rdata[2*i+1])
    cdata = np.fft.fft(cdata)
    hdata = postkernel(cdata, impose_hermitian)
    return hdata

def c2r_1d(input, nx, impose_hermitian=False):
    if impose_hermitian:
        hdata = symmetrize_1d(input, nx)
    else:
        hdata = input
    cdata = np.empty([nx], dtype=complex)
    for i in range(nx // 2 + 1):
        cdata[i] = hdata[i]
    for i in range(nx // 2 + 1, nx):
        cdata[i] = hdata[nx - i].conj()
    cdata = np.fft.ifft(cdata)
    return cdata.real

def prekernel(hdata, nx, impose_hermitian=False):
    cdata = np.empty(nx // 2, dtype=complex)
    I = complex(0, 1)
    p = hdata[0]
    q = hdata[nx // 2]
    if impose_hermitian:
        cdata[0] = complex(p.real + q.real,
                           p.real - q.real)
    else:
        cdata[0] = complex(p.real - p.imag + q.real + q.imag,
                           p.real + p.imag - q.real + q.imag)

        
    if nx % 4 == 0:
        cdata[nx // 4] =  hdata[nx // 4].conjugate() * 2
    for idx_p in range(1, (nx // 2 + 1) // 2):
        idx_q = nx // 2 - idx_p
        omegaNp = cmath.exp(2.0 * math.pi * I * idx_p / nx);
        omegaNq = -omegaNp.conjugate()
        p = hdata[idx_p]
        q = hdata[idx_q]
        cdata[idx_p] = p * (1 + I * omegaNp) + q.conjugate() * (1 - I * omegaNp)
        cdata[idx_q] = q * (1 + I * omegaNq) + p.conjugate() * (1 - I * omegaNq)
    return cdata

def c2r_1d_even(hdata, nx, impose_hermitian=False):
    #print(hdata[0])
    cdata = prekernel(hdata, nx, impose_hermitian)
    cdata = np.fft.ifft(cdata)
    rdata = np.empty(nx)
    for i in range(nx // 2):
        rdata[2 * i] = cdata[i].real
        rdata[2 * i + 1] = cdata[i].imag
    return rdata * 0.5
        
def r2c_2d(rdata, nx, ny):
    cdata = np.empty([nx, ny], dtype=complex)
    for i in range(nx):
        for j in range(ny):
            cdata[i][j] = rdata[i][j]
    cdata = np.fft.fft2(cdata)
    return cdata[0:nx,0:ny // 2 + 1]

def r2c_2d_decomp(rdata, nx, ny, impose_hermitian=False):
    hdata = np.empty([nx, ny // 2 + 1], dtype=complex)
    for i in range(nx):
        hdata[i] = r2c_1d(rdata[i], ny, impose_hermitian)
    for j in range(ny // 2 + 1):
        hdata[:,j] = np.fft.fft(hdata[:,j])
    return hdata

def r2c_2d_even(rdata, nx, ny, impose_hermitian=False):
    hdata = np.empty([nx, ny // 2 + 1], dtype=complex)
    for i in range(nx):
        hdata[i] = r2c_1d_even(rdata[i], ny, impose_hermitian)
    for j in range(ny // 2 + 1):
        hdata[:,j] = np.fft.fft(hdata[:,j])
    return hdata

def c2r_2d(hdata, nx, ny):
    cdata = np.zeros([nx, ny], dtype=complex)
    for i in range(nx):
        for j in range(ny // 2 + 1):
            cdata[i][j] = hdata[i][j];
    for i in [0, nx // 2]:
        for j in range(ny // 2 + 1, ny):
            cdata[i][j] = hdata[i][ny - j].conj();
    for i in range(1, nx // 2):
        for j in range(ny // 2 + 1, ny):
            cdata[i][j] = hdata[nx - i][ny - j].conj();
            cdata[nx - i][j] = hdata[i][ny - j].conj();
    cdata = np.fft.ifft2(cdata)
    return cdata.real
    
def c2r_2d_decomp(hdata, nx, ny, impose_hermitian=False):
    cdata = np.empty([nx, ny // 2 + 1], dtype=complex)
    for j in range(ny // 2 + 1):
        cdata[:,j] = np.fft.ifft(hdata[:,j])
    rdata = np.empty([nx, ny])
    for i in range(nx):
        rdata[i] = c2r_1d(cdata[i], ny, impose_hermitian)
    return rdata

def c2r_2d_even(hdata, nx, ny, impose_hermitian=False):
    cdata = np.empty([nx, ny // 2 + 1], dtype=complex)
    for j in range(ny // 2 + 1):
        cdata[:,j] = np.fft.ifft(hdata[:,j])
    rdata = np.empty([nx, ny])
    for i in range(nx):
        rdata[i] = c2r_1d_even(cdata[i], ny, impose_hermitian)
    return rdata

def r2c_3d(rdata, nx, ny, nz):
    cdata = np.empty([nx, ny, nz], dtype=complex)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cdata[i][j][k] = rdata[i][j][k]
    cdata = np.fft.fftn(cdata)
    return cdata[0:nx,0:ny,0:nz//2 + 1]

def r2c_3d_decomp(rdata, nx, ny, nz, impose_hermitian=False):
    hdata = np.empty([nx, ny, nz // 2 + 1], dtype=complex)
    for i in range(nx):
        for j in range(ny):
            hdata[i][j] = r2c_1d(rdata[i][j], nz, impose_hermitian)
    for i in range(nx):
        for k in range(nz // 2 + 1):
            hdata[i,:,k] = np.fft.fft(hdata[i,:,k])
    for j in range(ny):
        for k in range(nz // 2 + 1):
            hdata[:,j,k] = np.fft.fft(hdata[:,j,k])
    return hdata

def r2c_3d_even(rdata, nx, ny, nz, impose_hermitian=False):
    hdata = np.empty([nx, ny, nz // 2 + 1], dtype=complex)
    for i in range(nx):
        for j in range(ny):
            hdata[i][j] = r2c_1d_even(rdata[i][j], nz, impose_hermitian)
    for i in range(nx):
        for k in range(nz // 2 + 1):
            hdata[i,:,k] = np.fft.fft(hdata[i,:,k])
    for j in range(ny):
        for k in range(nz // 2 + 1):
            hdata[:,j,k] = np.fft.fft(hdata[:,j,k])
    return hdata


print()
print("checking our impose Hermitian code:")

print()
print("1D")
x = np.empty([nx])
for i in range(nx):
    x[i] = random.random()
X = np.fft.rfft(x)
#print(X)
print(is_symmetric_1d(X, nx))
print(np.allclose(X,  symmetrize_1d(X, nx)))


print()
print("2D")

x = np.empty([nx, ny])
for i in range(nx):
    for j in range(ny):
        x[i][j] = random.random()
X = np.fft.rfft2(x)
#print(X)
print(is_symmetric_2d(X, nx, ny))
print(np.allclose(X,  symmetrize_2d(X, nx, ny)))

        
Z = np.empty([nx, ny // 2 + 1], dtype=complex)
for i in range(nx):
    for j in range(ny //2 + 1):
        Z[i,j] = complex(random.random(), random.random())

#print(Z)
Z = symmetrize_2d(Z, nx, ny)
if is_symmetric_2d(Z, nx, ny):
    print("Z is symmetric")
else:
    print("Z isn't symmetric")
#print(Z)


print()
print("3D:", nx, ny, nz)
Z = np.empty([nx, ny, nz // 2 + 1], dtype=complex)
for i in range(nx):
    for j in range(ny):
        for k in range(nz // 2 + 1):
            Z[i,j,k] = complex(random.random(), random.random())
print(Z)
print("symmetrized:")
Z = symmetrize_3d(Z, nx, ny, nz)
print(Z)
print("Did symmetrize_3d work?", is_symmetric_3d(Z, nx, ny, nz))

x = np.empty([nx, ny, nz])
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            x[i,j,k] = random.random()
#X = r2c_3d_even(x, nx, ny, nz)
X = np.fft.rfftn(x)
print(X)
print("Is the r2c output symmetric?", is_symmetric_3d(X, nx, ny, nz))

print()
print("1D:", nx)

print()
print("direct")

x = np.empty([nx])
for i in range(nx):
    x[i] = random.random()
#print(x)
X = np.fft.rfft(x)
print("np.fft.rfft:")
#print(X)
print("embedded:")
X0 = r2c_1d(x, nx)
#print(X0)
print(np.allclose(X, X0))
print("embedded with imposed symmetry:")
X00 = r2c_1d(x, nx, True)
#print(X00)
print(np.allclose(X, X00))
print("even")
X000 = r2c_1d_even(x, nx)
#print(X000)
print(np.allclose(X, X000))



print()
print("inverse")
print("np.fft.irfft")
xx = np.fft.irfft(X, nx)
#print(xx)
print("embedded:")
xx0 = c2r_1d(X, nx)
#print(xx0)
print(np.allclose(xx, xx0))
print("embedded with Hermitian imposed:")
xx0 = c2r_1d(X, nx, True)
#print(xx0)
print(np.allclose(xx, xx0))
print("even")
xx0 = c2r_1d_even(X, nx)
#print(xx0)
print(np.allclose(xx, xx0))
print("even impose")
xx0 = c2r_1d_even(X, nx, True)
#print(xx0)
print(np.allclose(xx, xx0))


print()
print("2D:", nx, ny)

print()
print("direct")

x = np.empty([nx, ny])

for i in range(nx):
    for j in range(ny):
        x[i][j] = random.random()

print("np.fft.rfft2:")
X = np.fft.rfft2(x)
#print(X)
if is_symmetric_2d(X, nx, ny):
    print("X is symmetric")
else:
    print("X isn't symmetric")
print("2D embedded:")
X0 = r2c_2d(x, nx, ny)
#print(X0)
print(np.allclose(X, X0))
print("1D embedded:")
X00 = r2c_2d_decomp(x, nx, ny)
#print(X00)
print(np.allclose(X, X00))
print("1D embedded with 1D Hermitian imposed:")
X000 = r2c_2d_decomp(x, nx, ny, True)
#print(X000)
print(np.allclose(X, X000))
print("2D even:")
X000 = r2c_2d_even(x, nx, ny)
#print(X000)
print(np.allclose(X, X000))

print()
print("inverse")

print("np.fft.irfft2:")
xx = np.fft.irfft2(X, [nx, ny])
#print(xx)
print(np.allclose(x, xx))
print("2D embedded:")
xx0 = c2r_2d(X, nx, ny)
#print(xx0)
print(np.allclose(xx, xx0))
print("1D embedded:")
xx0 = c2r_2d_decomp(X, nx, ny)
#print(xx0)
print(np.allclose(xx, xx0))
print("1D embedded with 1D Hermitian imposed:")
xx0 = c2r_2d_decomp(X, nx, ny, True)
#print(xx0)
print(np.allclose(xx, xx0))
print("2D even:")
xx0 = c2r_2d_even(X, nx, ny)
#print(xx0)
print(np.allclose(xx, xx0))
print("2D even impose:")
xx0 = c2r_2d_even(X, nx, ny, True)
#print(xx0)
print(np.allclose(xx, xx0))


print()
print("3D:", nx, ny, nz)

x = np.empty([nx, ny, nz])
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            x[i][j][k] = random.random()
X = np.fft.rfftn(x)
X0 = r2c_3d(x, nx, ny, nz)
print(np.allclose(X,X0))
X0 = r2c_3d_decomp(x, nx, ny, nz)
print(np.allclose(X,X0))
X0 = r2c_3d_even(x, nx, ny, nz)
print(np.allclose(X,X0))



print()
print("1D inverse on malformed data:")
X = np.empty([nx], dtype=complex)
for i in range(nx):
    X[i] = complex(random.random(), random.random())
#print(X)
if is_symmetric_1d(X, nx):
    print("valid input")
else:
    print("invalid input")

print("np.fft.irfft")
xx = np.fft.irfft(X, nx)
#print(xx)
print("embedded:")
xx0 = c2r_1d(X, nx)
#print(xx0)
print(np.allclose(xx, xx0))
print("embedded with Hermitian imposed:")
xx0 = c2r_1d(X, nx, True)
#print(xx0)
print(np.allclose(xx, xx0))
print("even:")
xx0 = c2r_1d_even(X, nx)
#print(xx0)
print(np.allclose(xx, xx0))
print("even imposed:")
xx0 = c2r_1d_even(X, nx, True)
#print(xx0)
print(np.allclose(xx, xx0))


print()
print("1D inverse on malformed data:")
X = np.empty([nx], dtype=complex)
for i in range(nx):
    X[i] = complex(random.random(), random.random())
#print(X)
print("np.fft.irfft")
xx = np.fft.irfft(X, nx)
#print(xx)
print("embedded:")
xx0 = c2r_1d(X, nx)
#print(xx0)
print(np.allclose(xx, xx0))
print("embedded with Hermitian imposed:")
xx0 = c2r_1d(X, nx, True)
#print(xx0)
print(np.allclose(xx, xx0))
print("even")
xx0 = c2r_1d_even(X, nx)
#print(xx0)
print(np.allclose(xx, xx0))
print("even impose")
xx0 = c2r_1d_even(X, nx, True)
#print(xx0)
print(np.allclose(xx, xx0))



print()
print("2D inverse on malformed data:")
X = np.empty([nx, ny // 2 + 1], dtype=complex)
for i in range(nx):
    for j in range(ny // 2 + 1):
        X[i,j] = complex(random.random(), random.random())
print("np.fft.irfft2:")
xx = np.fft.irfft2(X, [nx, ny])
#print(xx)
print("2D embedded:")
xx0 = c2r_2d(X, nx, ny)
#print(xx0)
print(np.allclose(xx, xx0))
print("1D embedded:")
xx0 = c2r_2d_decomp(X, nx, ny)
#print(xx0)
print(np.allclose(xx, xx0))
print("1D embedded with 1D Hermitian imposed:")
xx0 = c2r_2d_decomp(X, nx, ny, True)
#print(xx0)
print(np.allclose(xx, xx0))
print("even:")
xx0 = c2r_2d_even(X, nx, ny)
#print(xx0)
print(np.allclose(xx, xx0))
print("even impose:")
xx0 = c2r_2d_even(X, nx, ny, True)
#print(xx0)
print(np.allclose(xx, xx0))



    
