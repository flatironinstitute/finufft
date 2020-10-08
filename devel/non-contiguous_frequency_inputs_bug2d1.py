# ahb updated for v2.0 interface, and complex c. Bug seems to have been fixed :)

import numpy as np

from finufft import nufft2d1

c = np.complex128(np.random.rand(2))

omega = np.arange(4).reshape((2, 2)) / 3 * np.pi

x0 = omega[:, 0]
y0 = omega[:, 1]

f0 = np.zeros((4, 4), order='F', dtype=np.complex128)

nufft2d1(x0, y0, c, f0.shape, out=f0, eps=1e-14)

x1 = x0.copy()
y1 = y0.copy()

f1 = np.zeros((4, 4), order='F', dtype=np.complex128)

nufft2d1(x1, y1, c, f1.shape, out=f1, eps=1e-14)

print('difference: %e' % np.linalg.norm(f0 - f1))
