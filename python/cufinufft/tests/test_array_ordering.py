import pytest

import numpy as np

import pycuda.autoinit # NOQA:401
import pycuda.gpuarray as gpuarray

from cufinufft import Plan

import utils


def test_type1_ordering(dtype=np.float32, shape=(16, 16, 16), M=4096, tol=1e-3):
    complex_dtype = utils._complex_dtype(dtype)

    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = gpuarray.to_gpu(k)
    c_gpu = gpuarray.to_gpu(c)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype)

    plan.setpts(*k_gpu)

    out_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)

    plan.execute(c_gpu, out=out_gpu)

    out_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype, order="F")

    with pytest.raises(TypeError, match="following requirement: C") as err:
        plan.execute(c_gpu, out=out_gpu)
