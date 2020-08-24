# demo of 1D type 1 FINUFFT in python. Should stay close to docs/python.rst
# Barnett 8/19/20

import numpy as np
import finufft
import time
np.random.seed(42)

# number of nonuniform points
M = 100000

# input nonuniform points
x = 2 * np.pi * np.random.uniform(size=M)

# their complex strengths
c = (np.random.standard_normal(size=M)
     + 1J * np.random.standard_normal(size=M))

# desired number of output Fourier modes
N = 1000000

# calculate the transform
t0 = time.time()
f = finufft.nufft1d1(x, c, N, eps=1e-9)
print("finufft1d1 done in {0:.2g} s.".format(time.time()-t0))

n = 142519   # do a math check, for a single output mode index n
assert((n>=-N/2.) & (n<N/2.))
ftest = sum(c * np.exp(1.j*n*x))
err = np.abs(f[n + N // 2] - ftest) / np.max(np.abs(f))
print("Error relative to max: {0:.2e}".format(err))
