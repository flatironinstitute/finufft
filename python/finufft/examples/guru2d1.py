# demo of vectorized 2D type 1 FINUFFT in python via guru interface. Should stay close to docs/python.rst
# Lu 8/20/20

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

# specify type 1 transform
nufft_type = 1

# instantiate the plan (note n_trans must be set here), also setting tolerance:
t0 = time.time()
plan = finufft.Plan(nufft_type, (N1, N2), eps=1e-9, n_trans=K)

# set the nonuniform points
plan.setpts(x, y)

# execute the plan (K transforms together, note c.shape must match)
f = plan.execute(c)
print("vectorized guru finufft2d1 done in {0:.2g} s.".format(time.time()-t0))

print(f.dtype)
print(f.shape)

k1 = 376     # do a math check, for a single output mode index (k1,k2)
k2 = -1000
t = K-2      # from the t'th transform
assert((k1>=-N1/2.) & (k1<N1/2.))   # float division easier here
assert((k2>=-N2/2.) & (k2<N2/2.))
assert((t>=0) & (t<K))
ftest = sum(c[t,:] * np.exp(1.j*(k1*x + k2*y)))
err = np.abs(f[t, k1+N1//2, k2+N2//2] - ftest) / np.max(np.abs(f))
print("Error relative to max: {0:.2e}".format(err))
