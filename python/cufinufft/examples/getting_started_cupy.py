import cupy as cp

import cufinufft

# number of nonuniform points
M = 100000

# grid size
N = 200000

# generate positions for the nonuniform points and the coefficients
x_gpu = 2 * cp.pi * cp.random.uniform(size=M)
c_gpu = cp.random.standard_normal(size=M) + 1j * cp.random.standard_normal(size=M)

# compute the transform
f_gpu = cufinufft.nufft1d1(x_gpu, c_gpu, (N,))

# move results off the GPU
f = f_gpu.get()
# check one output mode against the direct computation; asserts on the default tol
n = 1425
f_test = cp.sum(c_gpu * cp.exp(1j * n * x_gpu))
rel_err = cp.abs(f_gpu[n + N // 2] - f_test) / cp.max(cp.abs(f_gpu))
print(f"Relative error on mode {n} is {float(rel_err):.3g}")
assert rel_err < 10 * 1e-6  # default tol, as in example2d1_pycuda
