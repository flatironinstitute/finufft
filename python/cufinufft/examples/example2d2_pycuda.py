"""
Demonstrate the type 2 NUFFT using cuFINUFFT
"""

import numpy as np

import pycuda.autoinit
from pycuda.gpuarray import to_gpu

import cufinufft

# Set up parameters for problem.
N1, N2 = 37, 41                 # Size of uniform grid
M = 17                          # Number of nonuniform points
n_transf = 2                    # Number of input arrays
eps = 1e-6                      # Requested tolerance
dtype = np.float32              # Datatype (real)
complex_dtype = np.complex64    # Datatype (complex)

# Generate coordinates of non-uniform points.
x = np.random.uniform(-np.pi, np.pi, size=M)
y = np.random.uniform(-np.pi, np.pi, size=M)

# Generate grid values.
fk = (np.random.standard_normal((n_transf, N1, N2))
      + 1j * np.random.standard_normal((n_transf, N1, N2)))

# Cast to desired datatype.
x = x.astype(dtype)
y = y.astype(dtype)
fk = fk.astype(complex_dtype)

# Initialize the plan and set the points.
plan = cufinufft.Plan(2, (N1, N2), n_transf, eps=eps, dtype=complex_dtype)
plan.setpts(to_gpu(x), to_gpu(y))

# Execute the plan, reading from the uniform grid fk and storing the result
# in c_gpu.
c_gpu = plan.execute(to_gpu(fk))

# Retreive the result from the GPU.
c = c_gpu.get()

# Check accuracy of the transform at index jt.
jt = M // 2

for i in range(n_transf):
    # Calculate the true value of the type 2 transform at the index jt.
    m, n = np.mgrid[-(N1 // 2):(N1 + 1) // 2, -(N2 // 2):(N2 + 1) // 2]
    c_true = np.sum(fk[i] * np.exp(-1j * (m * x[jt] + n * y[jt])))

    # Calculate the absolute and relative error.
    err = np.abs(c[i, jt] - c_true)
    rel_err = err / np.max(np.abs(c[i]))

    print(f"[{i}] Absolute error on point [{jt}] is {err:.3g}")
    print(f"[{i}] Relative error on point [{jt}] is {rel_err:.3g}")

    assert(rel_err < 15 * eps)
