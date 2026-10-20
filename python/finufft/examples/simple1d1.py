# demo of 1D type 1 FINUFFT in python. Should stay close to docs/python.rst
# Barnett 8/19/20

# docs-start: simple1d1
import numpy as np
import finufft

# number of nonuniform points
M = 100000

# the nonuniform points
x = 2 * np.pi * np.random.uniform(size=M)

# their complex strengths
c = np.random.standard_normal(size=M) + 1j * np.random.standard_normal(size=M)

# desired number of Fourier modes (uniform outputs)
N = 200000

# calculate the transform
f = finufft.nufft1d1(x, c, N)
# docs-end: simple1d1

eps = 1e-6  # default tolerance
n = 71259  # do a math check, for a single output mode index n
assert -N / 2 <= n < N / 2
ftest = sum(c * np.exp(1.0j * n * x))
Fmax = np.max(np.abs(f))
if not np.isfinite(Fmax):
    raise SystemExit(f"FAILED: max |f| is not finite: {Fmax:.3g}")
err = np.abs(f[n + N // 2] - ftest) / Fmax
if err > 10 * eps:
    raise SystemExit(f"FAILED: relative error {err:.2e}, max |f| {Fmax:.3g}")
print(f"Error relative to max: {err:.2e}")
