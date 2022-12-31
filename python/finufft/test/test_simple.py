import pytest

import numpy as np
import finufft

def _get_real_dtype(dtype):
    return np.array(0, dtype=dtype).real.dtype

def _gen_pts(dim, n_pts, dtype):
    real_dtype = _get_real_dtype(dtype)
    pts =  2 * np.pi * np.random.rand(dim, n_pts).astype(real_dtype)
    return pts

def _gen_coefs(n_pts, n_tr, dtype):
    coefs = np.random.randn(*n_tr, n_pts) + 1j * np.random.randn(*n_tr, n_pts)
    coefs = coefs.astype(dtype)
    return coefs

def _gen_sig(n_modes, n_tr, dtype):
    sig = np.random.randn(*n_tr, *n_modes) + 1j * np.random.randn(*n_tr, *n_modes)
    sig = sig.astype(dtype)
    return sig

def _gen_sig_idx(n_modes, n_tr):
    idx = tuple(np.random.randint(0, n) for n in n_tr + n_modes)
    return idx

def _gen_coef_ind(n_pts, n_tr):
    ind = tuple(np.random.randint(0, n) for n in n_tr + (n_pts,))
    return ind

def _nudft1(pts, coefs, n_modes, idx):
    dtype = coefs.dtype
    dim = len(n_modes)

    _idx = (np.array(idx[-dim:]) - np.floor(np.array(n_modes) / 2)).astype(dtype)

    _coefs = coefs[idx[:-dim]]

    sig = np.sum(np.exp(1j * np.sum(_idx[:, np.newaxis] * pts, axis=0)) * _coefs)

    return sig

def _nudft2(pts, sig, ind):
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

def _nudft3(source_pts, source_coef, target_pts, ind):
    target_pt = target_pts[:, ind[-1]]
    target_pt = target_pt[:, np.newaxis]

    _source_coef = source_coef[ind[:-1]]

    target_coef = np.sum(np.exp(1j * np.sum(target_pt * source_pts, axis=0)) * _source_coef)

    return target_coef

@pytest.mark.parametrize("n_modes", [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)])
@pytest.mark.parametrize("n_pts", [10, 11])
@pytest.mark.parametrize("n_tr", [(), (2,)])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_out", [False, True])
def test_nufft1(n_modes, n_pts, n_tr, dtype, use_out):
    dim = len(n_modes)

    funs = {1: finufft.nufft1d1,
            2: finufft.nufft2d1,
            3: finufft.nufft3d1}

    fun = funs[dim]

    np.random.seed(0)

    pts = _gen_pts(dim, n_pts, dtype)
    coefs = _gen_coefs(n_pts, n_tr, dtype)
    idx = _gen_sig_idx(n_modes, n_tr)

    # See if it can handle square sizes from ints
    if all(n == n_modes[0] for n in n_modes):
        _n_modes = n_modes[0]
    else:
        _n_modes = n_modes

    if not use_out:
        sig = fun(*pts, coefs, _n_modes)
    else:
        sig = np.empty(n_tr + n_modes, dtype=dtype)
        fun(*pts, coefs, out=sig)

    sig0 = _nudft1(pts, coefs, n_modes, idx)

    assert sig.shape == n_tr + n_modes
    assert np.isclose(sig[idx], sig0)

@pytest.mark.parametrize("n_modes", [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)])
@pytest.mark.parametrize("n_pts", [10, 11])
@pytest.mark.parametrize("n_tr", [(), (2,)])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_out", [False, True])
def test_nufft2(n_modes, n_pts, n_tr, dtype, use_out):
    dim = len(n_modes)

    funs = {1: finufft.nufft1d2,
            2: finufft.nufft2d2,
            3: finufft.nufft3d2}

    fun = funs[dim]

    np.random.seed(0)

    pts = _gen_pts(dim, n_pts, dtype)
    sig = _gen_sig(n_modes, n_tr, dtype)
    ind = _gen_coef_ind(n_pts, n_tr)

    if not use_out:
        coef = fun(*pts, sig)
    else:
        coef = np.empty(n_tr + (n_pts,), dtype=dtype)
        fun(*pts, sig, out=coef)

    coef0 = _nudft2(pts, sig, ind)

    assert coef.shape == n_tr + (n_pts,)

    # NOTE: Tolerance should be tighter for n_tr != 1, but for now, loosen the
    # tolerance.
    if n_tr != ():
        rtol = 1e-4
    else:
        rtol = 1e-5

    assert np.isclose(coef[ind], coef0, rtol=rtol)

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("n_source_pts", [10, 11])
@pytest.mark.parametrize("n_target_pts", [10, 11])
@pytest.mark.parametrize("n_tr", [(), (2,)])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_out", [False, True])
def test_nufft3(dim, n_source_pts, n_target_pts, n_tr, dtype, use_out):
    funs = {1: finufft.nufft1d3,
            2: finufft.nufft2d3,
            3: finufft.nufft3d3}

    fun = funs[dim]

    np.random.seed(0)

    source_pts = _gen_pts(dim, n_source_pts, dtype)
    target_pts = _gen_pts(dim, n_target_pts, dtype)
    source_coefs = _gen_coefs(n_source_pts, n_tr, dtype)
    ind = _gen_coef_ind(n_target_pts, n_tr)

    if not use_out:
        target_coef = fun(*source_pts, source_coefs, *target_pts)
    else:
        target_coef = np.empty(n_tr + (n_target_pts,), dtype=dtype)
        fun(*source_pts, source_coefs, *target_pts, out=target_coef)

    target_coef0 = _nudft3(source_pts, source_coefs, target_pts, ind)

    assert target_coef.shape == n_tr + (n_target_pts,)

    # NOTE: Tolerance should be tighter for n_tr != 1, but for now, loosen the
    # tolerance.
    if n_tr != ():
        rtol = 1e-3
    else:
        rtol = 1e-4

    assert np.isclose(target_coef[ind], target_coef0, rtol=rtol)

def test_errors():
    with pytest.raises(RuntimeError, match="x dtype should be"):
        finufft.nufft1d1(np.zeros(1, "int64"), np.zeros(1), (4,))

    with pytest.raises(RuntimeError, match="n_modes must be"):
        finufft.nufft1d1(np.zeros(1), np.zeros(1), "")

    with pytest.raises(RuntimeError, match="n_modes dimension does not match"):
        finufft.nufft1d1(np.zeros(1), np.zeros(1), (2, 2))

    with pytest.raises(RuntimeError, match="elements of input n_modes must"):
        finufft.nufft1d1(np.zeros(1), np.zeros(1), (2.5,))

    with pytest.raises(RuntimeError, match="type 1 input must supply n_modes"):
        finufft.nufft1d1(np.zeros(1), np.zeros(1))

    with pytest.raises(RuntimeError, match="c.size must be divisible by x.size"):
        finufft.nufft1d1(np.zeros(2), np.zeros(3, np.complex128), 4)

    with pytest.raises(RuntimeError, match="type 2 input dimension must be either dim"):
        finufft.nufft1d2(np.zeros(1), np.zeros((2, 2, 2), np.complex128))

    with pytest.raises(RuntimeError, match="type 2 input dimension must be either dim"):
        finufft.nufft1d1(np.zeros(1), np.zeros(1, np.complex128), 4, out=np.zeros((2, 2, 4), np.complex128))

    with pytest.raises(RuntimeError, match="input n_trans and output n_trans do not match"):
        finufft.nufft1d1(np.zeros(1), np.zeros((3, 1), np.complex128), 4, out=np.zeros((2, 4), np.complex128))

    with pytest.raises(RuntimeError, match="input n_modes and output n_modes do not match"):
        finufft.nufft1d1(np.zeros(1), np.zeros(1, np.complex128), 4, out=np.zeros((3), np.complex128))
