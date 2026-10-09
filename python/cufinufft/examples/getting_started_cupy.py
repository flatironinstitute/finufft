# docs-start: getting-started-cupy
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
# docs-end: getting-started-cupy

# check: output finite and one mode against a direct sum
import numpy as np

tol = 1e-4
if not np.all(np.isfinite(f)):
    raise SystemExit(f"FAILED: max |f| is not finite: {np.max(np.abs(f)):.3g}")
k = N // 3
x = x_gpu.get()
c = c_gpu.get()
fk = np.sum(c * np.exp(1j * k * x))
err = abs(f[k + N // 2] - fk) / abs(f).max()
if err > 10 * tol:
    raise SystemExit(f"FAILED: rel err {err:.2e} exceeds 10*tol in mode {k}")
print(f"Error relative to max: {err:.2e}")
