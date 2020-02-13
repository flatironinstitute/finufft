# Simple 1d1 python interface call
# Lu 02/07/20.

import time
import pyfinufft as pf
import numpy as np

np.random.seed(42)

acc = 1.e-9
iflag = 1
N = int(1e6)
M = int(1e5)
x = np.random.uniform(-np.pi, np.pi, M)
c = np.random.randn(M) + 1.j * np.random.randn(M)
F = np.zeros([N], dtype=np.complex128)       # allocate F (modes out)
n_modes = np.ones([3], dtype=np.int64)
n_modes[0] = N

strt = time.time()

#opts
opts = pf.nufft_opts()
pf.pyfinufft_default_opts(opts)

#plan
plan = pf.finufft_plan()
status = pf.pyfinufft_makeplan(1,1,n_modes,iflag,1,acc,8,plan,opts)

#set pts
status = pf.pyfinufft_setpts(plan,M,x,None,None,0,None,None,None)

#exec
status = pf.pyfinufft_exec(plan,c,F)

#timing
print("Finished nufft in {0:.2g} seconds. Checking..."
      .format(time.time()-strt))

#check error
n = 142519      # mode to check
Ftest = 0.0
# this is so slow...
for j in range(M):
    Ftest += c[j] * np.exp(n * x[j] * 1.j)
Fmax = np.max(np.abs(F))
err = np.abs((F[n + N // 2] - Ftest) / Fmax)
print("Error relative to max of F: {0:.2e}".format(err))

#destroy
pf.pyfinufft_destroy(plan)
