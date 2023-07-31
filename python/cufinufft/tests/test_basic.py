import pytest

import numpy as np

import pycuda.autoinit # NOQA:401
import pycuda.gpuarray as gpuarray

from cufinufft import Plan

import utils

# NOTE: Tests below fail for tolerance 1e-4 (error executing plan).

DTYPES = [np.float32, np.float64]
SHAPES = [(16,), (16, 16), (16, 16, 16)]
MS = [256, 1024, 4096]
TOLS = [1e-2, 1e-3]
OUTPUT_ARGS = [False, True]

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_type1(dtype, shape, M, tol, output_arg):
    complex_dtype = utils._complex_dtype(dtype)

    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = gpuarray.to_gpu(k)
    c_gpu = gpuarray.to_gpu(c)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype)

    # Since k_gpu is an array of shape (dim, M), this will expand to
    # plan.setpts(k_gpu[0], ..., k_gpu[dim]), allowing us to handle all
    # dimensions with the same call.
    plan.setpts(*k_gpu)

    if output_arg:
        fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)
        plan.execute(c_gpu, out=fk_gpu)
    else:
        fk_gpu = plan.execute(c_gpu)

    fk = fk_gpu.get()

    utils.verify_type1(k, c, fk, tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_type2(dtype, shape, M, tol, output_arg):
    complex_dtype = utils._complex_dtype(dtype)

    k, fk = utils.type2_problem(dtype, shape, M)

    k_gpu = gpuarray.to_gpu(k)
    fk_gpu = gpuarray.to_gpu(fk)

    plan = Plan(2, shape, eps=tol, dtype=complex_dtype)

    plan.setpts(*k_gpu)

    if output_arg:
        c_gpu = gpuarray.GPUArray(shape=(M,), dtype=complex_dtype)
        plan.execute(fk_gpu, out=c_gpu)
    else:
        c_gpu = plan.execute(fk_gpu)

    c = c_gpu.get()

    utils.verify_type2(k, fk, c, tol)


def test_opts(shape=(8, 8, 8), M=32, tol=1e-3):
    dtype = np.float32

    complex_dtype = utils._complex_dtype(dtype)

    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = gpuarray.to_gpu(k)
    c_gpu = gpuarray.to_gpu(c)
    fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype, gpu_sort=False,
                     gpu_maxsubprobsize=10)

    plan.setpts(k_gpu[0], k_gpu[1], k_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    fk = fk_gpu.get()

    utils.verify_type1(k, c, fk, tol)

