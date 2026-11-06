# demo of vectorized 2D type 1 FINUFFT in single precision via guru interface.
# Should stay close to docs/python.rst
# Lu 8/20/20

import finufft
import numpy as np

np.random.seed(42)

M = 100000
x = 2 * np.pi * np.random.uniform(size=M)
y = 2 * np.pi * np.random.uniform(size=M)
K = 4
c = np.random.standard_normal(size=(K, M)) + 1j * np.random.standard_normal(size=(K, M))
N1 = 1000
N2 = 2000
nufft_type = 1

# docs-start: guru2d1f
# convert input data to single precision
x = x.astype("float32")
y = y.astype("float32")
c = c.astype("complex64")

# instantiate the plan and set the points
plan = finufft.Plan(nufft_type, (N1, N2), n_trans=K, dtype="complex64", eps=1e-3)
plan.setpts(x, y)

# execute the plan, giving single-precision output
f = plan.execute(c)
# docs-end: guru2d1f

assert f.dtype == np.complex64
assert f.shape == (K, N1, N2)

eps = 1e-3  # as passed to the plan above
k1 = 37  # do a math check, for a single output mode index (k1,k2)
k2 = -100
assert -N1 / 2 <= k1 < N1 / 2  # float division easier here
assert -N2 / 2 <= k2 < N2 / 2
ftest = c @ np.exp(1.0j * (k1 * x + k2 * y))
Fmax = np.max(np.abs(f), axis=(1, 2))
if not np.all(np.isfinite(Fmax)):
    raise SystemExit(f"FAILED: max |f| is not finite: {Fmax.max():.3g}")
err = np.max(np.abs(f[:, k1 + N1 // 2, k2 + N2 // 2] - ftest) / Fmax)
if err > 10 * eps:
    raise SystemExit(f"FAILED: max relative error {err:.2e}, max |f| {Fmax.max():.3g}")
print(f"Max error relative to max: {err:.2e}")
