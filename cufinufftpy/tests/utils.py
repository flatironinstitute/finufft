import numpy as np


def _complex_dtype(dtype):
    if dtype == np.float32:
        complex_dtype = np.complex64
    elif dtype == np.float64:
        complex_dtype = np.complex128
    else:
        raise TypeError("dtype should be np.float32 or np.float64.")

    return complex_dtype


def _real_dtype(complex_dtype):
    if complex_dtype == np.complex64:
        real_dtype = np.float32
    elif complex_dtype == np.complex128:
        real_dtype = np.float64
    else:
        raise TypeError("dtype should be np.complex64 or np.complex128.")

    return real_dtype


def gen_nu_pts(M, dim=3, seed=0):
    np.random.seed(seed)
    kxyz = np.random.uniform(-np.pi, np.pi, (dim, M))
    kxyz = kxyz.astype(np.float64)
    return kxyz


def gen_uniform_data(shape, seed=0):
    np.random.seed(seed)
    fk = np.random.standard_normal(shape + (2,))
    fk = fk.astype(np.float64).view(np.complex128)[..., 0]
    return fk


def gen_nonuniform_data(M, seed=0):
    np.random.seed(seed)
    c = np.random.standard_normal(2 * M)
    c = c.astype(np.float64).view(np.complex128)
    return c


def make_grid(shape):
    dim = len(shape)
    shape = (1,) * (3 - dim) + shape

    grids = [np.arange(-(N // 2), (N + 1) // 2) for N in shape]
    z, y, x = np.meshgrid(*grids, indexing='ij')
    return np.stack((x, y, z))


def direct_type1(c, kxyz, shape, ind):
    xyz = make_grid(shape)

    phase = kxyz.T @ xyz.reshape((3, -1))[:, ind]
    fk = np.sum(c * np.exp(1j * phase))

    return fk


def direct_type2(fk, kxyz):
    xyz = make_grid(fk.shape)

    phase = kxyz @ xyz.reshape((3, -1))
    c = np.sum(fk.ravel() * np.exp(-1j * phase))

    return c
