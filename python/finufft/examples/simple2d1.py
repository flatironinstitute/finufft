# demo of 2D type 1 FINUFFT in python. Should stay close to docs/python.rst
# Barnett 8/19/20

import numpy as np
import finufft
import time
np.random.seed(42)

# number of nonuniform points
M = 100000

# the nonuniform points in the square [0,2pi)^2
x = 2 * np.pi * np.random.uniform(size=M)
y = 2 * np.pi * np.random.uniform(size=M)

# their complex strengths
c = (np.random.standard_normal(size=M)
     + 1J * np.random.standard_normal(size=M))

# desired number of Fourier modes (in x,y directions respectively)
N1 = 1000
N2 = 2000

# calculate the transform
t0 = time.time()
f = finufft.nufft2d1(x, y, c, (N1,N2), eps=1e-9)
print("finufft2d1 done in {0:.2g} s.".format(time.time()-t0))

k1 = 376   # do a math check, for a single output mode index (k1,k2)
k2 = -1000
assert((k1>=-N1/2.) & (k1<N1/2.))   # float division easier here
assert((k2>=-N2/2.) & (k2<N2/2.))
ftest = sum(c * np.exp(1.j*(k1*x + k2*y)))
err = np.abs(f[k1+N1//2, k2+N2//2] - ftest) / np.max(np.abs(f))
print("Error relative to max: {0:.2e}".format(err))
