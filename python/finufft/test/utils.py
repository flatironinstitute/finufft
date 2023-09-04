import numpy as np


def _complex_dtype(dtype):
    dtype = np.dtype(dtype)

    if dtype == np.float32:
        complex_dtype = np.complex64
    elif dtype == np.float64:
        complex_dtype = np.complex128
    else:
        raise TypeError("dtype should be np.float32 or np.float64.")

    return complex_dtype


def _real_dtype(complex_dtype):
    complex_dtype = np.dtype(complex_dtype)

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


def gen_sig_idx(n_modes, n_tr):
    idx = tuple(np.random.randint(0, n) for n in n_tr + n_modes)
    return idx


def gen_coef_ind(n_pts, n_tr):
    ind = tuple(np.random.randint(0, n) for n in n_tr + (n_pts,))
    return ind


def type1_problem(dtype, shape, M, n_trans=()):
    real_dtype = _real_dtype(dtype)
    dim = len(shape)

    k = gen_nu_pts(M, dim=dim).astype(real_dtype)
    c = gen_nonuniform_data(M, n_trans=n_trans).astype(dtype)

    return k, c


def type2_problem(dtype, shape, M, n_trans=()):
    real_dtype = _real_dtype(dtype)
    dim = len(shape)

    k = gen_nu_pts(M, dim=dim).astype(real_dtype)
    fk = gen_uniform_data(n_trans + shape).astype(dtype)

    return k, fk

def type3_problem(dtype, dim, n_source_pts, n_target_pts, n_trans=()):
    real_dtype = _real_dtype(dtype)

    source_pts = gen_nu_pts(n_source_pts, dim=dim).astype(real_dtype)
    source_coefs = gen_nonuniform_data(n_source_pts, n_trans=n_trans).astype(dtype)
    target_pts = gen_nu_pts(n_target_pts, dim=dim).astype(real_dtype)

    return source_pts, source_coefs, target_pts


def make_grid(shape):
    dim = len(shape)
    shape = shape

    grids = [np.arange(-(N // 2), (N + 1) // 2) for N in shape]
    grids = np.meshgrid(*grids, indexing='ij')
    return np.stack(grids)


def direct_type1(pts, coefs, n_modes, idx):
    dtype = coefs.dtype
    dim = len(n_modes)

    _idx = (np.array(idx[-dim:]) - np.floor(np.array(n_modes) / 2)).astype(dtype)

    _coefs = coefs[idx[:-dim]]

    sig = np.sum(np.exp(1j * np.sum(_idx[:, np.newaxis] * pts, axis=0)) * _coefs)

    return sig


def direct_type2(pts, sig, ind):
    dtype = sig.dtype
    dim = pts.shape[0]
    n_modes = sig.shape[-dim:]

    grids = [slice(-np.floor(n / 2), np.ceil(n / 2)) for n in n_modes]
    grid = np.mgrid[grids].astype(dtype)

    pt = pts[:, ind[-1]]
    pt = pt.reshape((dim,) + dim * (1,))

    _sig = sig[ind[:-1]]

    coef = np.sum(np.exp(-1j * np.sum(pt * grid, axis=0)) * _sig)

    return coef


def direct_type3(source_pts, source_coefs, target_pts, ind):
    target_pt = target_pts[:, ind[-1]]
    target_pt = target_pt[:, np.newaxis]

    _source_coef = source_coefs[ind[:-1]]

    target_coef = np.sum(np.exp(1j * np.sum(target_pt * source_pts, axis=0)) * _source_coef)

    return target_coef


def verify_type1(pts, coefs, shape, sig_est, tol):
    dim = pts.shape[0]

    n_trans = coefs.shape[:-1]

    assert sig_est.shape[:-dim] == n_trans
    assert sig_est.shape[-dim:] == shape

    idx = gen_sig_idx(shape, n_trans)

    fk_est = sig_est[idx]
    fk_target = direct_type1(pts, coefs, shape, idx)

    type1_rel_err = np.linalg.norm(fk_target - fk_est) / np.linalg.norm(fk_target)

    assert type1_rel_err < 25 * tol


def verify_type2(pts, sig, coefs_est, tol):
    dim = pts.shape[0]

    n_trans = sig.shape[:-dim]
    n_pts = pts.shape[-1]

    assert coefs_est.shape == n_trans + (n_pts,)

    ind = gen_coef_ind(n_pts, n_trans)

    c_est = coefs_est[ind]
    c_target = direct_type2(pts, sig, ind)

    type2_rel_err = np.linalg.norm(c_target - c_est) / np.linalg.norm(c_target)

    assert type2_rel_err < 25 * tol


def verify_type3(source_pts, source_coef, target_pts, target_coef, tol):
    dim = source_pts.shape[0]

    n_source_pts = source_pts.shape[-1]
    n_target_pts = target_pts.shape[-1]
    n_tr = source_coef.shape[:-1]

    assert target_coef.shape == n_tr + (n_target_pts,)

    ind = gen_coef_ind(n_source_pts, n_tr)

    target_est = target_coef[ind]
    target_true = direct_type3(source_pts, source_coef, target_pts, ind)

    type3_rel_err = np.linalg.norm(target_est - target_true) / np.linalg.norm(target_true)

    assert type3_rel_err < 100 * tol
