import cupy as cp

import cufinufft

# number of nonuniform points
M = 100000

# grid size
N = 200000

# generate positions for the nonuniform points and the coefficients
x_gpu = 2 * cp.pi * cp.random.uniform(size=M)
c_gpu = (cp.random.standard_normal(size=M)
         + 1J * cp.random.standard_normal(size=M))

# create plan
plan = cufinufft.Plan(1, (N,), dtype=cp.complex128)

# set the nonuniform points
plan.setpts(x_gpu)

# execute the plan
f_gpu = plan.execute(c_gpu)

# move results off the GPU
f = f_gpu.get()
