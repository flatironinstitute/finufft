# docs-start: getting_started_cupy-import
import cupy as cp

import cufinufft
# docs-end: getting_started_cupy-import

# docs-start: getting_started_cupy-setup
# number of nonuniform points
M = 100000

# grid size
N = 200000

# generate positions for the nonuniform points and the coefficients
x_gpu = 2 * cp.pi * cp.random.uniform(size=M)
c_gpu = cp.random.standard_normal(size=M) + 1j * cp.random.standard_normal(size=M)

# docs-end: getting_started_cupy-setup

# docs-start: getting_started_cupy-transform
# compute the transform
f_gpu = cufinufft.nufft1d1(x_gpu, c_gpu, (N,))

# move results off the GPU
f = f_gpu.get()
# docs-end: getting_started_cupy-transform

# check one mode against a direct sum; the default tolerance is 1e-6
import numpy as np

tol = 1e-6
k = N // 3
fk = np.sum(c_gpu.get() * np.exp(1j * k * x_gpu.get()))
err = abs(f[k + N // 2] - fk) / np.max(np.abs(f))
print(f"rel err in mode {k} is {err:.3g}")
if not err <= 10 * tol:
    raise SystemExit("FAILED: accuracy gate")
