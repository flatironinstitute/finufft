import numpy as np

from finufftpy import nufft2d1

c = np.arange(2)

omega = np.arange(4).reshape((2, 2)) / 3 * np.pi

x0 = omega[:, 0]
y0 = omega[:, 1]

f0 = np.zeros((4, 4), order='F', dtype=np.complex128)

nufft2d1(x0, y0, c, 1, 1e-15, f0.shape[0], f0.shape[1], f0)

x1 = x0.copy()
y1 = y0.copy()

f1 = np.zeros((4, 4), order='F', dtype=np.complex128)

nufft2d1(x1, y1, c, 1, 1e-15, f1.shape[0], f1.shape[1], f1)

print('difference: %e' % np.linalg.norm(f0 - f1))
