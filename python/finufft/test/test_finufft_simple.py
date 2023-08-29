import pytest

import numpy as np
import finufft

import utils


@pytest.mark.parametrize("n_modes", [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)])
@pytest.mark.parametrize("n_pts", [10, 11])
@pytest.mark.parametrize("n_tr", [(), (2,)])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_out", [False, True])
def test_nufft1(n_modes, n_pts, n_tr, dtype, use_out):
    dim = len(n_modes)
    real_dtype = utils._real_dtype(dtype)

    funs = {1: finufft.nufft1d1,
            2: finufft.nufft2d1,
            3: finufft.nufft3d1}

    fun = funs[dim]

    pts, coefs = utils.type1_problem(real_dtype, n_modes, n_pts, n_trans=n_tr)

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

    utils.verify_type1(pts, coefs, sig, 1e-6)

@pytest.mark.parametrize("n_modes", [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)])
@pytest.mark.parametrize("n_pts", [10, 11])
@pytest.mark.parametrize("n_tr", [(), (2,)])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_out", [False, True])
def test_nufft2(n_modes, n_pts, n_tr, dtype, use_out):
    dim = len(n_modes)
    real_dtype = utils._real_dtype(dtype)

    funs = {1: finufft.nufft1d2,
            2: finufft.nufft2d2,
            3: finufft.nufft3d2}

    fun = funs[dim]

    pts, sig = utils.type2_problem(real_dtype, n_modes, n_pts, n_trans=n_tr)

    if not use_out:
        coef = fun(*pts, sig)
    else:
        coef = np.empty(n_tr + (n_pts,), dtype=dtype)
        fun(*pts, sig, out=coef)

    utils.verify_type2(pts, sig, coef, 1e-6)

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("n_source_pts", [10, 11])
@pytest.mark.parametrize("n_target_pts", [10, 11])
@pytest.mark.parametrize("n_tr", [(), (2,)])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("use_out", [False, True])
def test_nufft3(dim, n_source_pts, n_target_pts, n_tr, dtype, use_out):
    real_dtype = utils._real_dtype(dtype)

    funs = {1: finufft.nufft1d3,
            2: finufft.nufft2d3,
            3: finufft.nufft3d3}

    fun = funs[dim]

    source_pts, source_coefs, target_pts = utils.type3_problem(real_dtype,
            dim, n_source_pts, n_target_pts, n_trans=n_tr)

    if not use_out:
        target_coef = fun(*source_pts, source_coefs, *target_pts)
    else:
        target_coef = np.empty(n_tr + (n_target_pts,), dtype=dtype)
        fun(*source_pts, source_coefs, *target_pts, out=target_coef)

    utils.verify_type3(source_pts, source_coefs, target_pts, target_coef, 1e-6)

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
