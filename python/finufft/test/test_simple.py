import pytest

import numpy as np
import finufft

def _get_real_dtype(dtype):
    return np.array(0, dtype=dtype).real.dtype

def _gen_pts(dim, n_pts, dtype):
    real_dtype = _get_real_dtype(dtype)
    pts =  2 * np.pi * np.random.rand(dim, n_pts).astype(real_dtype)
    return pts

def _gen_coefs(n_pts, dtype):
    coefs = np.random.randn(n_pts) + 1j * np.random.randn(n_pts)
    coefs = coefs.astype(dtype)
    return coefs

def _gen_sig(n_modes, dtype):
    sig = np.random.randn(*n_modes) + 1j * np.random.randn(*n_modes)
    sig = sig.astype(dtype)
    return sig

def _gen_sig_idx(n_modes):
    idx = tuple(np.random.randint(0, n) for n in n_modes)
    return idx

def _gen_coef_ind(n_pts):
    ind = np.random.randint(0, n_pts)
    return ind

def _nudft1(pts, coefs, n_modes, idx):
    dtype = coefs.dtype
    dim = len(n_modes)

    idx = (np.array(idx) - np.floor(np.array(n_modes) / 2)).astype(dtype)

    sig = np.sum(np.exp(1j * np.sum(idx[:, np.newaxis] * pts, axis=0)) * coefs)

    return sig

def _nudft2(pts, sig, ind):
    dtype = sig.dtype
    n_modes = sig.shape
    dim = len(n_modes)

    grids = [slice(-np.floor(n / 2), np.ceil(n / 2)) for n in n_modes]
    grid = np.mgrid[grids].astype(dtype)

    pt = pts[:, ind]
    pt = pt.reshape((dim,) + dim * (1,))

    coef = np.sum(np.exp(-1j * np.sum(pt * grid, axis=0)) * sig)

    return coef

def _nudft3(source_pts, source_coef, target_pts, ind):
    target_pt = target_pts[:, ind]
    target_pt = target_pt[:, np.newaxis]

    target_coef = np.sum(np.exp(1j * np.sum(target_pt * source_pts, axis=0)) * source_coef)

    return target_coef

@pytest.mark.parametrize("n_modes", [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)])
@pytest.mark.parametrize("n_pts", [10, 11])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_nufft1(n_modes, n_pts, dtype):
    dim = len(n_modes)

    funs = {1: finufft.nufft1d1,
            2: finufft.nufft2d1,
            3: finufft.nufft3d1}

    fun = funs[dim]

    np.random.seed(0)

    pts = _gen_pts(dim, n_pts, dtype)
    coefs = _gen_coefs(n_pts, dtype)
    idx = _gen_sig_idx(n_modes)

    sig = fun(*pts, coefs, n_modes)

    sig0 = _nudft1(pts, coefs, n_modes, idx)

    assert sig.shape == n_modes
    assert np.isclose(sig[idx], sig0)

@pytest.mark.parametrize("n_modes", [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)])
@pytest.mark.parametrize("n_pts", [10, 11])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_nufft2(n_modes, n_pts, dtype):
    dim = len(n_modes)

    funs = {1: finufft.nufft1d2,
            2: finufft.nufft2d2,
            3: finufft.nufft3d2}

    fun = funs[dim]

    np.random.seed(0)

    pts = _gen_pts(dim, n_pts, dtype)
    sig = _gen_sig(n_modes, dtype)
    ind = _gen_coef_ind(n_pts)

    coef = fun(*pts, sig)

    coef0 = _nudft2(pts, sig, ind)

    assert coef.shape == (n_pts,)
    assert np.isclose(coef[ind], coef0)

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("n_source_pts", [10, 11])
@pytest.mark.parametrize("n_target_pts", [10, 11])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_nufft3(dim, n_source_pts, n_target_pts, dtype):
    funs = {1: finufft.nufft1d3,
            2: finufft.nufft2d3,
            3: finufft.nufft3d3}

    fun = funs[dim]

    np.random.seed(0)

    source_pts = _gen_pts(dim, n_source_pts, dtype)
    target_pts = _gen_pts(dim, n_target_pts, dtype)
    source_coefs = _gen_coefs(n_source_pts, dtype)
    ind = _gen_coef_ind(n_target_pts)

    target_coef = fun(*source_pts, source_coefs, *target_pts)

    target_coef0 = _nudft3(source_pts, source_coefs, target_pts, ind)

    assert target_coef.shape == (n_target_pts,)
    assert np.isclose(target_coef[ind], target_coef0, rtol=1e-4)
