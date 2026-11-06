# demo of vectorized 2D type 1 FINUFFT in python via guru interface.
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

# docs-start: guru2d1
# specify type 1 transform
nufft_type = 1

# instantiate the plan (note ntrans must be set here)
plan = finufft.Plan(nufft_type, (N1, N2), n_trans=K)

# set the nonuniform points
plan.setpts(x, y)

# execute the plan
f = plan.execute(c)
# docs-end: guru2d1

assert f.shape == (K, N1, N2)

eps = 1e-6  # default tolerance
k1 = 376  # do a math check, for a single output mode index (k1,k2)
k2 = -1000
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
