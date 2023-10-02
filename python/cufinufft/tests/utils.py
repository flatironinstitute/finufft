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


def gen_nonuniform_data(M, seed=0, n_trans=()):
    np.random.seed(seed)
    c = np.random.standard_normal(2 * M * int(np.prod(n_trans)))
    c = c.astype(np.float64).view(np.complex128)
    c = c.reshape(n_trans + (M,))
    return c


def type1_problem(dtype, shape, M, n_trans=()):
    complex_dtype = _complex_dtype(dtype)
    dim = len(shape)

    k = gen_nu_pts(M, dim=dim).astype(dtype)
    c = gen_nonuniform_data(M, n_trans=n_trans).astype(complex_dtype)

    return k, c


def type2_problem(dtype, shape, M, n_trans=()):
    complex_dtype = _complex_dtype(dtype)
    dim = len(shape)

    k = gen_nu_pts(M, dim=dim).astype(dtype)
    fk = gen_uniform_data(n_trans + shape).astype(complex_dtype)

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
    fk = np.sum(c * np.exp(1j * phase), -1)

    return fk


def direct_type2(fk, k, dim):
    grid = make_grid(fk.shape[-dim:])

    phase = k @ grid.reshape((dim, -1))

    # Ravel the spatial dimensions only.
    fk = fk.reshape(fk.shape[:-dim] + (np.prod(fk.shape[-dim:]),))
    c = np.sum(fk * np.exp(-1j * phase), -1)

    return c


def verify_type1(k, c, fk, tol):
    dim = fk.ndim - (c.ndim - 1)

    shape = fk.shape[-dim:]

    ind = int(0.1789 * np.prod(shape))

    # Ravel the spatial dimensions only.
    fk_est = fk.reshape(fk.shape[:-dim] + (np.prod(fk.shape[-dim:]),))[..., ind]
    fk_target = direct_type1(c, k, shape, ind)

    type1_rel_err = np.linalg.norm(fk_target - fk_est) / np.linalg.norm(fk_target)

    assert type1_rel_err < 25 * tol


def verify_type2(k, fk, c, tol):
    dim = fk.ndim - (c.ndim - 1)

    M = c.shape[-1]

    ind = M // 2

    c_est = c[..., ind]
    c_target = direct_type2(fk, k[:, ind], dim)

    type2_rel_err = np.linalg.norm(c_target - c_est) / np.linalg.norm(c_target)

    assert type2_rel_err < 25 * tol


def transfer_funcs(module_name):
    if module_name == "pycuda":
        import pycuda.autoinit # NOQA:401
        from pycuda.gpuarray import to_gpu
        def to_cpu(obj):
            return obj.get()
    elif module_name == "cupy":
        import cupy
        def to_gpu(obj):
            return cupy.array(obj)
        def to_cpu(obj):
            return obj.get()
    elif module_name == "numba":
        import numba.cuda
        to_gpu = numba.cuda.to_device
        def to_cpu(obj):
            return obj.copy_to_host()
    elif module_name == "torch":
        import torch
        def to_gpu(obj):
            return torch.as_tensor(obj, device=torch.device("cuda"))
        def to_cpu(obj):
            return obj.cpu().numpy()
    else:
        raise TypeError(f"Unsupported framework: {module_name}")

    return to_gpu, to_cpu
