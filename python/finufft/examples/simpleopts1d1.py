# demo of 1D type 1 FINUFFT options (modeord, upsampfac, out) in python.
# Barnett 10/25/17. Added upsampfac, 6/18/18

import finufft
import numpy as np

np.random.seed(42)

# docs-start: simpleopts1d1
tol = 1.0e-9
N = int(1e6)
M = int(1e5)
x = np.random.uniform(-np.pi, np.pi, M)
c = np.random.randn(M) + 1.0j * np.random.randn(M)

# default options
F1 = finufft.nufft1d1(x, c, N, tol=tol, isign=1)

# FFT mode order, written into the preallocated output array
F2 = np.zeros(N, dtype=np.complex128)
Ftest2 = finufft.nufft1d1(x, c, out=F2, tol=tol, isign=1, modeord=1)

# lower upsampling factor (sigma)
F3 = finufft.nufft1d1(x, c, N, tol=tol, isign=1, upsampfac=1.25)
# docs-end: simpleopts1d1

if Ftest2 is not F2:
    raise SystemExit("FAILED: out=F2 not used, returned a different array")

n = 142519  # mode to check
Ftest = np.sum(c * np.exp(1.0j * n * x))
# modeord=1 gives FFT mode order; there mode n sits at index n
for F, i in ((F1, n + N // 2), (F2, n), (F3, n + N // 2)):
    Fmax = np.max(np.abs(F))
    if not np.isfinite(Fmax):
        raise SystemExit(f"FAILED: max |F| is not finite: {Fmax:.3g}")
    err = np.abs(F[i] - Ftest) / Fmax
    if err > 10 * tol:
        raise SystemExit(f"FAILED: relative error {err:.2e}, max |F| {Fmax:.3g}")
print(f"Error relative to max: {err:.2e}")
