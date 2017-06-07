import time
import finufft
import numpy as np

np.random.seed(42)

acc = 1.e-9
N = int(1e6)
M = int(1e6)
x = np.random.uniform(-np.pi, np.pi, N)
c = np.random.randn(N) + 1.j * np.random.randn(N)

strt = time.time()
F = finufft.nufft1d1(x, c, M, acc, 1)
print("Finished nufft in {0:.2e} seconds. Checking..."
      .format(time.time()-strt))

n = 142519
Ftest = 0.0
for j in range(M):
    Ftest += c[j] * np.exp(n * x[j] * 1.j)
Fmax = np.max(np.abs(F));
err = np.abs((F[n + N // 2] - Ftest) / Fmax)
print("Error relative to max of F: {0}".format(err))
