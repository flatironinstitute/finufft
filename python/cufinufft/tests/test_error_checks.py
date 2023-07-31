import numpy as np
import pytest

import pycuda.autoinit # NOQA:401
import pycuda.gpuarray as gpuarray

from cufinufft import Plan

import utils


def test_set_nu_raises_on_dtype():
    dtype = np.complex64

    M = 4096
    tol = 1e-3
    shape = (16, 16, 16)
    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=dim).astype(dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)

    # Here we'll intentionally contruct an incorrect array dtype.
    kxyz_gpu_wrong_type = gpuarray.to_gpu(kxyz.real.astype(np.float64))

    plan = Plan(1, shape, eps=tol, dtype=dtype)

    with pytest.raises(TypeError):
        plan.setpts(kxyz_gpu_wrong_type[0],
                     kxyz_gpu[1], kxyz_gpu[2])
    with pytest.raises(TypeError):
        plan.setpts(kxyz_gpu[0],
                     kxyz_gpu_wrong_type[1], kxyz_gpu[2])
    with pytest.raises(TypeError):
        plan.setpts(kxyz_gpu[0],
                     kxyz_gpu[1], kxyz_gpu_wrong_type[2])
    with pytest.raises(TypeError):
        plan.setpts(kxyz_gpu_wrong_type[0],
                     kxyz_gpu_wrong_type[1], kxyz_gpu_wrong_type[2])


def test_set_pts_raises_on_size():
    dtype = np.float32
    complex_dtype = np.complex64

    M = 8
    tol = 1e-3
    shape = (16, 16, 16)
    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=dim).astype(dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype)

    with pytest.raises(TypeError, match="`y` must be of shape") as err:
        plan.setpts(kxyz_gpu[0], kxyz_gpu[1][:4])

    with pytest.raises(TypeError, match="`z` must be of shape") as err:
        plan.setpts(kxyz_gpu[0], kxyz_gpu[1], kxyz_gpu[2][:4])


def test_set_pts_raises_on_nonvector():
    dtype = np.float32
    complex_dtype = np.complex64

    M = 8
    tol = 1e-3
    shape = (16, 16, 16)
    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=dim).astype(dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype)

    with pytest.raises(TypeError, match="`x` must be a vector") as err:
        plan.setpts(kxyz)


def test_set_pts_raises_on_number_of_args():
    dtype = np.float32
    complex_dtype = np.complex64

    M = 8
    tol = 1e-3
    shape = (16,)
    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=3).astype(dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype)

    with pytest.raises(TypeError, match="is 1, but `y` was specified") as err:
        plan.setpts(*kxyz_gpu[:2])

    shape = (16, 16)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype)

    with pytest.raises(TypeError, match="is 2, but `z` was specified") as err:
        plan.setpts(*kxyz_gpu)


def test_wrong_field_names():
    with pytest.raises(TypeError, match="Invalid option 'foo'") as err:
        plan = Plan(1, (8, 8), foo="bar")


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

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype)

    plan.setpts(kxyz_gpu[0],
                 kxyz_gpu[1], kxyz_gpu[2])

    with pytest.raises(TypeError):
        plan.execute(c_gpu, fk_gpu_wrong_dtype)

    with pytest.raises(TypeError):
        plan.execute(c_gpu_wrong_dtype, fk_gpu)


def test_dtype_errors():
    with pytest.raises(TypeError, match="Expected complex64 or complex128") as err:
        Plan(1, (8, 8), dtype="uint8")


def test_dtype_warnings():
    with pytest.warns(DeprecationWarning, match="Converting to complex64") as record:
        Plan(1, (8, 8), dtype="float32")

    with pytest.warns(DeprecationWarning, match="Converting to complex128") as record:
        Plan(1, (8, 8), dtype="float64")
