import cufinufft

import torch

# number of nonuniform points
M = 100000

# grid size
N = 200000

# generate positions for the nonuniform points and the coefficients
x_gpu = 2 * torch.pi * torch.rand(size=(M,)).cuda()
c_gpu = (torch.randn(size=(M,)) + 1J * torch.randn(size=(M,))).cuda()

# compute the transform
f_gpu = cufinufft.nufft1d1(x_gpu, c_gpu, (N,))

# move results off the GPU
f = f_gpu.cpu()
