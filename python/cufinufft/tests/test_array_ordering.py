import pytest

import numpy as np

import pycuda.autoinit # NOQA:401
import pycuda.gpuarray as gpuarray

from cufinufft import Plan

import utils

def test_type2_ordering(dtype=np.float32, shape=(16, 16, 16), M=4096, tol=1e-3):
    complex_dtype = utils._complex_dtype(dtype)

    k = utils.gen_nu_pts(M).astype(dtype)
    fk = utils.gen_uniform_data(shape).astype(complex_dtype)

    fkTT = fk.T.copy().T

    k_gpu = gpuarray.to_gpu(k)
    fk_gpu = gpuarray.to_gpu(fk)
    fkTT_gpu = gpuarray.to_gpu(fkTT)

    plan = Plan(2, shape, eps=tol, dtype=complex_dtype)

    plan.setpts(k_gpu[0], k_gpu[1], k_gpu[2])

    c_gpu = plan.execute(fk_gpu)

    with pytest.raises(TypeError) as err:
        cTT_gpu = plan.execute(fkTT_gpu)
    assert "following requirement: C" in err.value.args[0]

    # Ideally, it should be possible to get this to align with true output,
    # but corrently does not look like it.

    # c = c_gpu.get()
    # cTT = cTT_gpu.get()

    # assert np.allclose(c, cTT, rtol=1e-2)


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

    with pytest.raises(TypeError) as err:
        plan.execute(c_gpu, out=out_gpu)
    assert "following requirement: C" in err.value.args[0]
