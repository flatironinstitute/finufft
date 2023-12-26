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

# move the data to GPU
x_gpu = to_gpu(x)
c_gpu = to_gpu(c)

# compute the transform
f_gpu = cufinufft.nufft1d1(x_gpu, c_gpu, (N,))

# move results off the GPU
f = f_gpu.get()
