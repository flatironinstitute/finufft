import numpy as np

import pycuda.autoinit
from pycuda.gpuarray import GPUArray, to_gpu

from cufinufft import cufinufft

# number of nonuniform points
M = 100000

# grid size
N = 200000

# generate positions for the nonuniform points and the coefficients
x = 2 * np.pi * np.random.uniform(size=M)
c = (np.random.standard_normal(size=M)
     + 1J * np.random.standard_normal(size=M))

# create plan
plan = cufinufft(1, (N,), dtype=np.float64)

# set the nonuniform points
plan.set_pts(to_gpu(x))

# allocate output array
f_gpu = GPUArray((N,), dtype=np.complex128)

# execute the plan
plan.execute(to_gpu(c), f_gpu)

# move results off the GPU
f = f_gpu.get()
