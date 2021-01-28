"""
Demonstrate the type 2 NUFFT using cuFINUFFT
"""

import numpy as np

import pycuda.autoinit
from pycuda.gpuarray import GPUArray, to_gpu

from cufinufft import cufinufft

# Set up parameters for problem.
N1, N2 = 37, 41                 # Size of uniform grid
M = 17                          # Number of nonuniform points
n_transf = 2                    # Number of input arrays
eps = 1e-6                      # Requested tolerance
dtype = np.float32              # Datatype (real)
complex_dtype = np.complex64    # Datatype (complex)

# Generate coordinates of non-uniform points.
kx = np.random.uniform(-np.pi, np.pi, size=M)
ky = np.random.uniform(-np.pi, np.pi, size=M)

# Generate grid values.
fk = (np.random.standard_normal((n_transf, N1, N2))
      + 1j * np.random.standard_normal((n_transf, N1, N2)))

# Cast to desired datatype.
kx = kx.astype(dtype)
ky = ky.astype(dtype)
fk = fk.astype(complex_dtype)

# Allocate memory for the nonuniform coefficients on the GPU.
c_gpu = GPUArray((n_transf, M), dtype=complex_dtype)

# Initialize the plan and set the points.
plan = cufinufft(2, (N1, N2), n_transf, eps=eps, dtype=dtype)
plan.set_pts(to_gpu(kx), to_gpu(ky))

# Execute the plan, reading from the uniform grid fk c and storing the result
# in c_gpu.
plan.execute(c_gpu, to_gpu(fk))

# Retreive the result from the GPU.
c = c_gpu.get()

# Check accuracy of the transform at index jt.
jt = M // 2

for i in range(n_transf):
    # Calculate the true value of the type 2 transform at the index jt.
    x, y = np.mgrid[-(N1 // 2):(N1 + 1) // 2, -(N2 // 2):(N2 + 1) // 2]
    c_true = np.sum(fk[i] * np.exp(-1j * (x * kx[jt] + y * ky[jt])))

    # Calculate the absolute and relative error.
    err = np.abs(c[i, jt] - c_true)
    rel_err = err / np.max(np.abs(c[i]))

    print(f"[{i}] Absolute error on point [{jt}] is {err:.3g}")
    print(f"[{i}] Relative error on point [{jt}] is {rel_err:.3g}")

    assert(rel_err < 10 * eps)
