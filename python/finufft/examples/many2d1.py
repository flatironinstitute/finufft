# demo of vectorized 2D type 1 FINUFFT in python. Should stay close to docs/python.rst
# Barnett 8/19/20

import finufft
import numpy as np

np.random.seed(42)

# number of nonuniform points
M = 100000

# the nonuniform points in the square [0,2pi)^2
x = 2 * np.pi * np.random.uniform(size=M)
y = 2 * np.pi * np.random.uniform(size=M)

# docs-start: many2d1
# number of transforms
K = 4

# generate K stacked strength arrays
c = np.random.standard_normal(size=(K, M)) + 1j * np.random.standard_normal(size=(K, M))

# desired number of Fourier modes (in x,y directions respectively)
N1 = 1000
N2 = 2000

# calculate the K transforms simultaneously (K is inferred from c.shape)
tol = 1e-9
f = finufft.nufft2d1(x, y, c, (N1, N2), tol=tol)
# docs-end: many2d1
assert f.shape == (K, N1, N2)

k1 = 376  # do a math check, for a single output mode index (k1,k2)
k2 = -1000
assert -N1 / 2 <= k1 < N1 / 2  # float division easier here
assert -N2 / 2 <= k2 < N2 / 2
ftest = c @ np.exp(1.0j * (k1 * x + k2 * y))
Fmax = np.max(np.abs(f), axis=(1, 2))
if not np.all(np.isfinite(Fmax)):
    raise SystemExit(f"FAILED: max |f| is not finite: {Fmax.max():.3g}")
err = np.max(np.abs(f[:, k1 + N1 // 2, k2 + N2 // 2] - ftest) / Fmax)
if err > 10 * tol:
    raise SystemExit(f"FAILED: max relative error {err:.2e}, max |f| {Fmax.max():.3g}")
print(f"Max error relative to max: {err:.2e}")
