# torch is imported first on purpose: this used to break the cufinufft import
# on the CI image (#410). Keep the order.
import torch

import cufinufft

# number of nonuniform points
M = 100000

# grid size
N = 200000

# generate positions for the nonuniform points and the coefficients
x_gpu = 2 * torch.pi * torch.rand(size=(M,)).cuda()
c_gpu = (torch.randn(size=(M,)) + 1j * torch.randn(size=(M,))).cuda()

# compute the transform
f_gpu = cufinufft.nufft1d1(x_gpu, c_gpu, (N,))

# move results off the GPU
f = f_gpu.cpu()
# check one output mode against the direct computation; asserts on the default tol
n = 1425
f_test = torch.sum(c_gpu * torch.exp(1j * n * x_gpu))
rel_err = torch.abs(f_gpu[n + N // 2] - f_test) / torch.max(torch.abs(f_gpu))
print(f"Relative error on mode {n} is {float(rel_err):.3g}")
assert rel_err < 10 * 1e-6  # default tol, as in example2d1_pycuda
