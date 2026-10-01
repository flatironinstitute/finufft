# Simple 1d1 python interface call
# Lu 02/07/20.

import time
import finufft as fp
import numpy as np

np.random.seed(42)

# single precision cannot resolve better than about max(N_i) * eps_mach, so a
# single-precision plan needs both a modest mode count and a coarse tolerance.
# Same sizes as examples/guru1d1f.cpp.
N = int(1e4)
M = int(1e5)
tol = 1e-3
x = np.random.uniform(-np.pi, np.pi, M)
x = x.astype("float32")
c = np.random.randn(M) + 1.0j * np.random.randn(M)
c = c.astype("complex64")
F = np.zeros([N], dtype=np.complex64)  # allocate F (modes out)

strt = time.time()

# plan, using proper specifier for single-precision transform
plan = fp.Plan(1, (N,), eps=tol, dtype="complex64")

# set pts
plan.setpts(x)

# exec
plan.execute(c, F)

# timing
print("Finished nufft in {0:.2g} seconds. Checking...".format(time.time() - strt))

# check error
n = 143  # mode to check
Ftest = 0.0
# this is so slow...
for j in range(M):
    Ftest += c[j] * np.exp(n * x[j] * 1.0j)
assert np.all(np.isfinite(F))
Fmax = np.max(np.abs(F))
err = np.abs((F[n + N // 2] - Ftest) / Fmax)
print("Error relative to max of F: {0:.2e}".format(err))
assert err <= 10 * tol
