import pytest

import numpy as np

import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import cufinufft

import utils

DTYPES = [np.float32, np.float64]
SHAPES = [(64,), (64, 64), (64, 64, 64)]
MS = [256, 1024, 4096]
TOLS = [1e-2, 1e-3]
OUTPUT_ARGS = [False, True]

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_simple_type1(dtype, shape, M, tol, output_arg):
    real_dtype = dtype
    complex_dtype = utils._complex_dtype(dtype)

    dim = len(shape)

    fun = {1: cufinufft.nufft1d1,
           2: cufinufft.nufft2d1,
           3: cufinufft.nufft3d1}[dim]

    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = gpuarray.to_gpu(k)
    c_gpu = gpuarray.to_gpu(c)

    if output_arg:
        fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)
        fun(*k_gpu, c_gpu, out=fk_gpu, eps=tol)
    else:
        fk_gpu = fun(*k_gpu, c_gpu, shape, eps=tol)

    fk = fk_gpu.get()

    utils.verify_type1(k, c, fk, tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_simple_type2(dtype, shape, M, tol, output_arg):
    real_dtype = dtype
    complex_dtype = utils._complex_dtype(dtype)

    dim = len(shape)

    fun = {1: cufinufft.nufft1d2,
           2: cufinufft.nufft2d2,
           3: cufinufft.nufft3d2}[dim]

    k, fk = utils.type2_problem(dtype, shape, M)

    k_gpu = gpuarray.to_gpu(k)
    fk_gpu = gpuarray.to_gpu(fk)

    if output_arg:
        c_gpu = gpuarray.GPUArray((M,), dtype=complex_dtype)
        fun(*k_gpu, fk_gpu, out=c_gpu)
    else:
        c_gpu = fun(*k_gpu, fk_gpu)

    c = c_gpu.get()

    utils.verify_type2(k, fk, c, tol)
