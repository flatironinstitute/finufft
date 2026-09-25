import numpy as np

import pycuda.autoinit
from pycuda.gpuarray import to_gpu

import cufinufft

# number of nonuniform points
M = 100000

# grid size
N = 200000

# generate positions for the nonuniform points and the coefficients
x = 2 * np.pi * np.random.uniform(size=M)
c = np.random.standard_normal(size=M) + 1j * np.random.standard_normal(size=M)

# move the data to GPU
x_gpu = to_gpu(x)
c_gpu = to_gpu(c)

# compute the transform
f_gpu = cufinufft.nufft1d1(x_gpu, c_gpu, (N,))

# move results off the GPU
f = f_gpu.get()
# check one output mode against the direct computation; asserts on the default tol
n = 1425
f_test = np.sum(c * np.exp(1j * n * x))
rel_err = np.abs(f[n + N // 2] - f_test) / np.max(np.abs(f))
print(f"Relative error on mode {n} is {float(rel_err):.3g}")
assert rel_err < 10 * 1e-6  # default tol, as in example2d1_pycuda
