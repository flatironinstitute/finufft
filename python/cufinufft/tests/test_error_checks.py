import numpy as np
import pytest

import pycuda.autoinit # NOQA:401
import pycuda.gpuarray as gpuarray

from cufinufft import cufinufft

import utils


def test_set_nu_raises_on_dtype():
    dtype = np.float32

    M = 4096
    tol = 1e-3
    shape = (16, 16, 16)
    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=dim).astype(dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)

    # Here we'll intentionally contruct an incorrect array dtype.
    kxyz_gpu_wrong_type = gpuarray.to_gpu(kxyz.astype(np.float64))

    plan = cufinufft(1, shape, eps=tol, dtype=dtype)

    with pytest.raises(TypeError):
        plan.set_pts(kxyz_gpu_wrong_type[0],
                     kxyz_gpu[1], kxyz_gpu[2])
    with pytest.raises(TypeError):
        plan.set_pts(kxyz_gpu[0],
                     kxyz_gpu_wrong_type[1], kxyz_gpu[2])
    with pytest.raises(TypeError):
        plan.set_pts(kxyz_gpu[0],
                     kxyz_gpu[1], kxyz_gpu_wrong_type[2])
    with pytest.raises(TypeError):
        plan.set_pts(kxyz_gpu_wrong_type[0],
                     kxyz_gpu_wrong_type[1], kxyz_gpu_wrong_type[2])


def test_set_pts_raises_on_size():
    dtype = np.float32

    M = 8
    tol = 1e-3
    shape = (16, 16, 16)
    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=dim).astype(dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)

    plan = cufinufft(1, shape, eps=tol, dtype=dtype)

    with pytest.raises(TypeError) as err:
        plan.set_pts(kxyz_gpu[0], kxyz_gpu[1][:4])
    assert 'kx and ky must be equal' in err.value.args[0]

    with pytest.raises(TypeError) as err:
        plan.set_pts(kxyz_gpu[0], kxyz_gpu[1], kxyz_gpu[2][:4])
    assert 'kx and kz must be equal' in err.value.args[0]


def test_wrong_field_names():
    with pytest.raises(TypeError) as err:
        plan = cufinufft(1, (8, 8), foo="bar")
    assert "Invalid option 'foo'" in err.value.args[0]


def test_exec_raises_on_dtype():
    dtype = np.float32
    complex_dtype = np.complex64

    M = 4096
    tol = 1e-3
    shape = (16, 16, 16)
    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=dim).astype(dtype)
    c = utils.gen_nonuniform_data(M).astype(complex_dtype)
    c_gpu = gpuarray.to_gpu(c)
    # Using c.real gives us wrong dtype here...
    c_gpu_wrong_dtype = gpuarray.to_gpu(c.real)

    kxyz_gpu = gpuarray.to_gpu(kxyz)
    fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)
    # Here we'll intentionally contruct an incorrect array dtype.
    fk_gpu_wrong_dtype = gpuarray.GPUArray(shape, dtype=np.complex128)

    plan = cufinufft(1, shape, eps=tol, dtype=dtype)

    plan.set_pts(kxyz_gpu[0],
                 kxyz_gpu[1], kxyz_gpu[2])

    with pytest.raises(TypeError):
        plan.execute(c_gpu, fk_gpu_wrong_dtype)

    with pytest.raises(TypeError):
        plan.execute(c_gpu_wrong_dtype, fk_gpu)
