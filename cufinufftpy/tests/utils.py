import numpy as np


def gen_nu_pts(M, seed=0):
    np.random.seed(seed)
    kxyz = np.random.uniform(-np.pi, np.pi, (3, M))
    kxyz = kxyz.astype(np.float32)
    return kxyz


def gen_uniform_data(shape, seed=0):
    np.random.seed(seed)
    fk = np.random.standard_normal(shape + (2,))
    fk = fk.astype(np.float32).view(np.complex64)[..., 0]
    return fk


def gen_nonuniform_data(M, seed=0):
    np.random.seed(seed)
    c = np.random.standard_normal(2 * M)
    c = c.astype(np.float32).view(np.complex64)
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
