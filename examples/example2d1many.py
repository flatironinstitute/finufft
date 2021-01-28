"""
Demonstrate the type 1 NUFFT using cuFINUFFT
"""

import numpy as np

import pycuda.autoinit
from pycuda.gpuarray import GPUArray, to_gpu

from cufinufft import cufinufft

# Set up parameters for problem.
N1, N2 = 59, 61                 # Size of uniform grid
M = 100                         # Number of nonuniform points
n_transf = 2                    # Number of input arrays
eps = 1e-6                      # Requested tolerance
dtype = np.float32              # Datatype (real)
complex_dtype = np.complex64    # Datatype (complex)

# Generate coordinates of non-uniform points.
kx = np.random.uniform(-np.pi, np.pi, size=M)
ky = np.random.uniform(-np.pi, np.pi, size=M)

# Generate source strengths.
c = (np.random.standard_normal((n_transf, M))
     + 1j * np.random.standard_normal((n_transf, M)))

# Cast to desired datatype.
kx = kx.astype(dtype)
ky = ky.astype(dtype)
c = c.astype(complex_dtype)

# Allocate memory for the uniform grid on the GPU.
fk_gpu = GPUArray((n_transf, N1, N2), dtype=complex_dtype)

# Initialize the plan and set the points.
plan = cufinufft(1, (N1, N2), n_transf, eps=eps, dtype=dtype)
plan.set_pts(to_gpu(kx), to_gpu(ky))

# Execute the plan, reading from the strengths array c and storing the
# result in fk_gpu.
plan.execute(to_gpu(c), fk_gpu)

# Retreive the result from the GPU.
fk = fk_gpu.get()

# Check accuracy of the transform at position (nt1, nt2).
nt1 = int(0.37 * N1)
nt2 = int(0.26 * N2)

for i in range(n_transf):
    # Calculate the true value of the type 1 transform at the uniform grid
    # point (nt1, nt2), which corresponds to the coordinate nt1 - N1 // 2 and
    # nt2 - N2 // 2.
    x, y = nt1 - N1 // 2, nt2 - N2 // 2
    fk_true = np.sum(c[i] * np.exp(1j * (x * kx + y * ky)))

    # Calculate the absolute and relative error.
    err = np.abs(fk[i, nt1, nt2] - fk_true)
    rel_err = err / np.max(np.abs(fk[i]))

    print(f"[{i}] Absolute error on mode [{nt1}, {nt2}] is {err:.3g}")
    print(f"[{i}] Relative error on mode [{nt1}, {nt2}] is {rel_err:.3g}")

    assert(rel_err < 10 * eps)
