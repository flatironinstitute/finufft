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
    k = np.random.uniform(-np.pi, np.pi, (dim, M))
    k = k.astype(np.float64)
    return k


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


def type1_problem(dtype, shape, M):
    complex_dtype = _complex_dtype(dtype)
    dim = len(shape)

    k = gen_nu_pts(M, dim=dim).astype(dtype)
    c = gen_nonuniform_data(M).astype(complex_dtype)

    return k, c


def type2_problem(dtype, shape, M):
    complex_dtype = _complex_dtype(dtype)
    dim = len(shape)

    k = gen_nu_pts(M, dim=dim).astype(dtype)
    fk = gen_uniform_data(shape).astype(complex_dtype)

    return k, fk


def make_grid(shape):
    dim = len(shape)
    shape = shape

    grids = [np.arange(-(N // 2), (N + 1) // 2) for N in shape]
    grids = np.meshgrid(*grids, indexing='ij')
    return np.stack(grids)


def direct_type1(c, k, shape, ind):
    dim = len(shape)

    grid = make_grid(shape)

    phase = k.T @ grid.reshape((dim, -1))[:, ind]
    fk = np.sum(c * np.exp(1j * phase))

    return fk


def direct_type2(fk, k):
    dim = fk.ndim

    grid = make_grid(fk.shape)

    phase = k @ grid.reshape((dim, -1))
    c = np.sum(fk.ravel() * np.exp(-1j * phase))

    return c
