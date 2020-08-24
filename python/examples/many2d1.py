# demo of vectorized 2D type 1 FINUFFT in python. Should stay close to docs/python.rst
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

# number of transforms
K = 4

# generate K stacked strength arrays
c = (np.random.standard_normal(size=(K, M))
     + 1J * np.random.standard_normal(size=(K, M)))

# desired number of Fourier modes (in x,y directions respectively)
N1 = 1000
N2 = 2000

# calculate the K transforms simultaneously (K is inferred from c.shape)
t0 = time.time()
f = finufft.nufft2d1(x, y, c, (N1,N2), eps=1e-9)
print("vectorized finufft2d1 done in {0:.2g} s.".format(time.time()-t0))
print(f.shape)

k1 = 376     # do a math check, for a single output mode index (k1,k2)
k2 = -1000
t = K-1      # from the t'th transform
assert((k1>=-N1/2.) & (k1<N1/2.))   # float division easier here
assert((k2>=-N2/2.) & (k2<N2/2.))
assert((t>=0) & (t<K))
ftest = sum(c[t,:] * np.exp(1.j*(k1*x + k2*y)))
err = np.abs(f[t, k1+N1//2, k2+N2//2] - ftest) / np.max(np.abs(f))
print("Error relative to max: {0:.2e}".format(err))
