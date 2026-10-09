# demo of 1D type 1 FINUFFT in single precision via the plan (guru) interface
# Lu 02/07/20.

import finufft
import numpy as np

np.random.seed(42)

N = int(1e4)
M = int(1e5)
tol = 1e-3
x = np.random.uniform(-np.pi, np.pi, M)
x = x.astype("float32")
c = np.random.randn(M) + 1.0j * np.random.randn(M)
c = c.astype("complex64")
F = np.zeros([N], dtype=np.complex64)  # allocate F (modes out)
n_modes = np.ones([1], dtype=np.int64)
n_modes[0] = N

# plan, using proper specifier for single-precision transform
plan = finufft.Plan(1, (N,), tol=tol, dtype="complex64")

# set pts
plan.setpts(x)

# exec
plan.execute(c, F)

# check error
n = 143  # mode to check
Ftest = np.sum(c * np.exp(1.0j * n * x))
Fmax = np.max(np.abs(F))
if not np.isfinite(Fmax):
    raise SystemExit(f"FAILED: max |F| is not finite: {Fmax:.3g}")
err = np.abs(F[n + N // 2] - Ftest) / Fmax
if err > 10 * tol:
    raise SystemExit(f"FAILED: relative error {err:.2e}, max |F| {Fmax:.3g}")
print(f"Error relative to max: {err:.2e}")
