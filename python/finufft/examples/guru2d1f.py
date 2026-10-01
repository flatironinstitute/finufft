# docs-start: guru2d1f
# demo of vectorized 2D type 1 FINUFFT in single precision via guru interface.
# Should stay close to docs/python.rst
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
c = np.random.standard_normal(size=(K, M)) + 1j * np.random.standard_normal(size=(K, M))

# convert input data to single precision
x = x.astype("float32")
y = y.astype("float32")
c = c.astype("complex64")

# desired number of Fourier modes (in x,y directions respectively)
N1 = 1000
N2 = 2000

# specify type 1 transform
nufft_type = 1

# instantiate the plan (note n_trans and dtype must be set here):
t0 = time.time()
# single precision resolves no finer than about max(N_i) * eps_mach, so eps=1e-4 fails
eps = 1e-3
plan = finufft.Plan(nufft_type, (N1, N2), eps=eps, n_trans=K, dtype="complex64")

# set the nonuniform points
plan.setpts(x, y)

# execute the plan, giving single-precision output
f = plan.execute(c)
print(
    "vectorized guru single-prec finufft2d1 done in {0:.2g} s.".format(time.time() - t0)
)

print(f.dtype)
print(f.shape)
# docs-end: guru2d1f

k1 = 37  # do a math check, for a single output mode index (k1,k2), every transform
k2 = -100
assert (k1 >= -N1 / 2.0) & (k1 < N1 / 2.0)  # float division easier here
assert (k2 >= -N2 / 2.0) & (k2 < N2 / 2.0)
for t in range(K):
    ftest = sum(c[t, :] * np.exp(1.0j * (k1 * x + k2 * y)))
    assert np.all(np.isfinite(f[t]))
    err = np.abs(f[t, k1 + N1 // 2, k2 + N2 // 2] - ftest) / np.max(np.abs(f[t]))
    print("Transform {0}, error relative to max: {1:.2e}".format(t, err))
    assert err <= 10 * eps
