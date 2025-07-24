import pytest

import numpy as np

import cufinufft
from cufinufft import _compat

import utils

DTYPES = [np.complex64, np.complex128]
SHAPES = [(16,), (16, 16), (16, 16, 16)]
N_TRANS = [(), (1,), (2,)]
MS = [256, 1024, 4096]
TOLS = [1e-3, 1e-6]
OUTPUT_ARGS = [False, True]

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_trans", N_TRANS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_simple_type1(to_gpu, to_cpu, dtype, shape, n_trans, M, tol, output_arg):
    dim = len(shape)

    # Select which function to call based on dimension.
    fun = {1: cufinufft.nufft1d1,
           2: cufinufft.nufft2d1,
           3: cufinufft.nufft3d1}[dim]

    k, c = utils.type1_problem(dtype, shape, M, n_trans=n_trans)

    k_gpu = to_gpu(k)
    c_gpu = to_gpu(c)

    if output_arg:
        # Ensure that output array has proper shape i.e., (N1, ...) for no
        # batch, (1, N1, ...) for batch of size one, and (n, N1, ...) for
        # batch of size n.
        fk_gpu = _compat.array_empty_like(c_gpu, n_trans + shape,
                dtype=dtype)

        fun(*k_gpu, c_gpu, out=fk_gpu, eps=tol)
    else:
        fk_gpu = fun(*k_gpu, c_gpu, shape, eps=tol)

    fk = to_cpu(fk_gpu)

    utils.verify_type1(k, c, fk, tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_trans", N_TRANS)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_simple_type2(to_gpu, to_cpu, dtype, shape, n_trans, M, tol, output_arg):
    dim = len(shape)

    fun = {1: cufinufft.nufft1d2,
           2: cufinufft.nufft2d2,
           3: cufinufft.nufft3d2}[dim]

    k, fk = utils.type2_problem(dtype, shape, M, n_trans=n_trans)

    k_gpu = to_gpu(k)
    fk_gpu = to_gpu(fk)

    if output_arg:
        c_gpu = _compat.array_empty_like(fk_gpu, n_trans + (M,),
                dtype=dtype)

        fun(*k_gpu, fk_gpu, eps=tol, out=c_gpu)
    else:
        c_gpu = fun(*k_gpu, fk_gpu, eps=tol)

    c = to_cpu(c_gpu)

    utils.verify_type2(k, fk, c, tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", list(set(len(shape) for shape in SHAPES)))
@pytest.mark.parametrize("n_source_pts", MS)
@pytest.mark.parametrize("n_target_pts", MS)
@pytest.mark.parametrize("n_trans", N_TRANS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_cufinufft3_simple(to_gpu, to_cpu, dtype, dim, n_source_pts, n_target_pts, n_trans, tol, output_arg):

    fun = {1: cufinufft.nufft1d3,
           2: cufinufft.nufft2d3,
           3: cufinufft.nufft3d3}[dim]

    source_pts, source_coefs, target_pts = utils.type3_problem(
        dtype, dim, n_source_pts, n_target_pts, n_trans
    )


    source_pts_gpu = to_gpu(source_pts)
    source_coefs_gpu = to_gpu(source_coefs)
    target_pts_gpu = to_gpu(target_pts)

    if output_arg:
        target_coefs_gpu = _compat.array_empty_like(
            source_coefs_gpu, n_trans + (n_target_pts,), dtype=dtype)

        fun(*source_pts_gpu, source_coefs_gpu, *target_pts_gpu, out=target_coefs_gpu, eps=tol)
    else:
        target_coefs_gpu = fun(*source_pts_gpu, source_coefs_gpu, *target_pts_gpu, eps=tol)

    target_coefs = to_cpu(target_coefs_gpu)

    utils.verify_type3(source_pts, source_coefs, target_pts, target_coefs, tol)
