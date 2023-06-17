import numpy as np

import pycuda.autoinit
from pycuda.gpuarray import to_gpu

import cufinufft

# number of nonuniform points
M = 100000

# grid size
N = 200000

# generate positions for the nonuniform points and the coefficients
x = 2 * np.pi * np.random.uniform(size=M)
c = (np.random.standard_normal(size=M)
     + 1J * np.random.standard_normal(size=M))

# create plan
plan = cufinufft.Plan(1, (N,), dtype=np.complex128)

# set the nonuniform points
plan.setpts(to_gpu(x))

# execute the plan
f_gpu = plan.execute(to_gpu(c))

# move results off the GPU
f = f_gpu.get()
