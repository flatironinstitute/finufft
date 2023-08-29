import pytest

import numpy as np
import finufft

import utils


SHAPES = [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)]
N_PTS = [10, 11]
N_TRANS = [(), (2,)]
DTYPES = [np.complex64, np.complex128]
OUTPUT_ARGS = [False, True]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_pts", N_PTS)
@pytest.mark.parametrize("n_trans", N_TRANS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_finufft1_simple(dtype, shape, n_pts, n_trans, output_arg):
    dim = len(shape)

    funs = {1: finufft.nufft1d1,
            2: finufft.nufft2d1,
            3: finufft.nufft3d1}

    fun = funs[dim]

    pts, coefs = utils.type1_problem(dtype, shape, n_pts, n_trans)

    # See if it can handle square sizes from ints
    if all(n == shape[0] for n in shape):
        _shape = shape[0]
    else:
        _shape = shape

    if not output_arg:
        sig = fun(*pts, coefs, _shape)
    else:
        sig = np.empty(n_trans + shape, dtype=dtype)
        fun(*pts, coefs, out=sig)

    utils.verify_type1(pts, coefs, shape, sig, 1e-6)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_pts", N_PTS)
@pytest.mark.parametrize("n_trans", N_TRANS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_finufft2_simple(dtype, shape, n_pts, n_trans, output_arg):
    dim = len(shape)

    funs = {1: finufft.nufft1d2,
            2: finufft.nufft2d2,
            3: finufft.nufft3d2}

    fun = funs[dim]

    pts, sig = utils.type2_problem(dtype, shape, n_pts, n_trans)

    if not output_arg:
        coefs = fun(*pts, sig)
    else:
        coefs = np.empty(n_trans + (n_pts,), dtype=dtype)
        fun(*pts, sig, out=coefs)

    utils.verify_type2(pts, sig, coefs, 1e-6)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", list(set(len(shape) for shape in SHAPES)))
@pytest.mark.parametrize("n_source_pts", N_PTS)
@pytest.mark.parametrize("n_target_pts", N_PTS)
@pytest.mark.parametrize("n_trans", N_TRANS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_finufft3_simple(dtype, dim, n_source_pts, n_target_pts, n_trans, output_arg):
    funs = {1: finufft.nufft1d3,
            2: finufft.nufft2d3,
            3: finufft.nufft3d3}

    fun = funs[dim]

    source_pts, source_coefs, target_pts = utils.type3_problem(dtype,
            dim, n_source_pts, n_target_pts, n_trans)

    if not output_arg:
        target_coefs = fun(*source_pts, source_coefs, *target_pts)
    else:
        target_coefs = np.empty(n_trans + (n_target_pts,), dtype=dtype)
        fun(*source_pts, source_coefs, *target_pts, out=target_coefs)

    utils.verify_type3(source_pts, source_coefs, target_pts, target_coefs, 1e-6)

def test_finufft_simple_errors():
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
