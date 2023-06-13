# convert DFM's simple demo to JFM interface, include modeord test.
# Barnett 10/25/17. Adde upsampfac, 6/18/18

import time
import finufft
import numpy as np

# print finufft.nufft1d1.__doc__

np.random.seed(42)

acc = 1.e-9
iflag = 1
N = int(1e6)
M = int(1e5)
x = np.random.uniform(-np.pi, np.pi, M)
c = np.random.randn(M) + 1.j * np.random.randn(M)
F = np.zeros([N], dtype=np.complex128)       # allocate F (modes out)

strt = time.time()
F = finufft.nufft1d1(x, c, N, eps=acc, isign=iflag, debug=1, spread_debug=1)
print("Finished nufft in {0:.2g} seconds. Checking..."
      .format(time.time()-strt))

n = 142519      # mode to check
Ftest = 0.0
# this is so slow...
for j in range(M):
    Ftest += c[j] * np.exp(n * x[j] * 1.j)
Fmax = np.max(np.abs(F))
err = np.abs((F[n + N // 2] - Ftest) / Fmax)
print("Error relative to max of F: {0:.2e}".format(err))

# now test FFT mode output version, overwriting F...
strt = time.time()
finufft.nufft1d1(x, c, out=F, eps=acc, isign=iflag, modeord=1)
print("Finished nufft in {0:.2g} seconds (modeord=1)"
      .format(time.time()-strt))
err = np.abs((F[n] - Ftest) / Fmax)   # now zero offset in F array
print("Error relative to max of F: {0:.2e}".format(err))

# now test low-upsampfac (sigma) version...
strt = time.time()
Ftest2 = finufft.nufft1d1(x, c, N, F, acc, iflag, upsampfac=1.25)
print(Ftest2 is F)
print("Finished nufft in {0:.2g} seconds (upsampfac=1.25)"
      .format(time.time()-strt))
err = np.abs((Ftest2[n + N // 2] - Ftest) / Fmax)   # now zero offset in F array
print("Error relative to max of F: {0:.2e}".format(err))
