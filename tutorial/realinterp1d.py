# demo 1D NUFFT interpolation of real periodic func *without* length-doubling.
# A real function is sampled on the 0-offset regular periodic grid, and
# we want to use the type-2 NUFFT to spectrally interpolate it efficiently.
# Barnett 8/13/25 for Kaya Unalmis.
import numpy as np
import finufft as fi

N = 5   # num samples
# a generic real test fun  with bandlimit <= (N-1)/2 so interp exact...
fun = lambda t: 1.0 + np.sin(t+1) + np.sin(2*t-2)    # bandlimit is 2
Nt = 100      # test targs
targs = np.random.rand(Nt)*2*np.pi
Nf = N//2 + 1    # num freq outputs for rfft; note rfft2 seems different :(
g = np.linspace(0,2*np.pi,N,endpoint=False)    # sample grid
f = fun(g)
c = (1/N) * np.fft.rfft(f)   # gets coeffs 0,1,..Nf-1  (don't forget prefac)
assert c.size==Nf

# Do the naive (double-length c array) NUFFT version:
cref = np.concatenate([np.conj(np.flip(c[1:])), c])   # reflect to 1-Nf...Nf-1 coeffs
ft = np.real(fi.nufft1d2(targs,cref,eps=1e-12,isign=1))    # f at targs (isign!)
# (taking Re here was just a formality; it is already real to eps_mach)
print("naive (reflected) 1d2 max err:", np.linalg.norm(fun(targs) - ft, np.inf))

# now demo avoid doubling the NUFFT length via freq shift and mult by phase:
c[1:] *= 2.0     # since each nonzero coeff appears twice in reflected array
N0 = Nf//2       # starting freq shift that FINUFFT interprets the c array
ftp = fi.nufft1d2(targs,c,eps=1e-12,isign=1)   # f at targs but with phase error
# the key step: rephase (to account for shift), only then take Re (needed!)...
ft = np.real( ftp * (np.cos(N0*targs) + 1j*np.sin(N0*targs)))   # guess 1j sign
print("unpadded 1d2 max err:", np.linalg.norm(fun(targs) - ft, np.inf))
