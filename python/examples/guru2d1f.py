# demo of vectorized 2D type 1 FINUFFT in python for guru interface. Should stay close to docs/python.rst
# Lu 8/20/20

import numpy as np
import finufftpy
import time
np.random.seed(42)

# number of nonuniform points
M = 100000

# the nonuniform points in the square [0,2pi)^2
x = 2 * np.pi * np.random.uniform(size=M)
y = 2 * np.pi * np.random.uniform(size=M)

# number of transforms
K = 7

# generate K stacked strength arrays
c = (np.random.standard_normal(size=(K, M))
     + 1J * np.random.standard_normal(size=(K, M)))

# convert input data to single precision
x = x.astype('float32')
y = y.astype('float32')
c = c.astype('complex64')

# desired number of Fourier modes (in x,y directions respectively)
N1 = 1000
N2 = 2000

# guru interface calls to calculate the K transforms simultaneously
t0 = time.time()

# specify type 1 transform
nufft_type = 1

# instantiate the plan (note ntrans must be set here)
plan = finufftpy.Plan(nufft_type, (N1, N2), n_trans=K, dtype='float32')

# set the nonuniform points
plan.setpts(x, y)

# execute the plan
f = plan.execute(c)
print(f.dtype)

print("vectorized finufft2d1 done in {0:.2g} s.".format(time.time()-t0))
print(f.shape)

k1 = 37     # do a math check, for a single output mode index (k1,k2)
k2 = -100
t = K-2      # from the t'th transform
assert((k1>=-N1/2.) & (k1<N1/2.))   # float division easier here
assert((k2>=-N2/2.) & (k2<N2/2.))
assert((t>=0) & (t<K))
ftest = sum(c[t,:] * np.exp(1.j*(k1*x + k2*y)))
err = np.abs(f[t, k1+N1//2, k2+N2//2] - ftest) / np.max(np.abs(f))
print("Error relative to max: {0:.2e}".format(err))
