# demo of 1D type 1 FINUFFT in python via the plan (guru) interface
# Lu 02/07/20.

import finufft
import numpy as np

np.random.seed(42)

N = int(1e6)
M = int(1e5)
x = np.random.uniform(-np.pi, np.pi, M)
c = np.random.randn(M) + 1.0j * np.random.randn(M)
F = np.zeros([N], dtype=np.complex128)  # allocate F (modes out)
n_modes = np.ones([1], dtype=np.int64)
n_modes[0] = N

tol = 1e-6

# plan
plan = finufft.Plan(1, (N,), tol=tol)

# set pts
plan.setpts(x)

# exec
plan.execute(c, F)

# check error
n = 142519  # mode to check
Ftest = np.sum(c * np.exp(1.0j * n * x))
Fmax = np.max(np.abs(F))
if not np.isfinite(Fmax):
    raise SystemExit(f"FAILED: max |F| is not finite: {Fmax:.3g}")
err = np.abs(F[n + N // 2] - Ftest) / Fmax
if err > 10 * tol:
    raise SystemExit(f"FAILED: relative error {err:.2e}, max |F| {Fmax:.3g}")
print(f"Error relative to max: {err:.2e}")
