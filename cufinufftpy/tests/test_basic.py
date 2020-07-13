import numpy as np
import pytest

import pycuda.autoinit # NOQA:401
import pycuda.gpuarray as gpuarray

from cufinufftpy import cufinufft

import utils


def _test_type1(dtype, shape=(16, 16, 16), M=4096, tol=1e-3):
    complex_dtype = utils._complex_dtype(dtype)

    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=dim).astype(dtype)
    c = utils.gen_nonuniform_data(M).astype(complex_dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)
    c_gpu = gpuarray.to_gpu(c)
    fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)

    plan = cufinufft(1, shape, 1, tol, dtype=dtype)

    plan.set_nu_pts(M, kxyz_gpu[0], kxyz_gpu[1], kxyz_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    fk = fk_gpu.get()

    ind = int(0.1789 * np.prod(shape))

    fk_est = fk.ravel()[ind]
    fk_target = utils.direct_type1(c, kxyz, shape, ind)

    type1_rel_err = np.abs(fk_target - fk_est) / np.abs(fk_target)

    plan.destroy()

    print('Type 1 relative error:', type1_rel_err)

    assert type1_rel_err < 0.01


def test_type1_32(shape=(16, 16, 16), M=4096, tol=1e-3):
    return _test_type1(dtype=np.float32, shape=shape, M=M, tol=tol)


def test_type1_64(shape=(16, 16, 16), M=4096, tol=1e-3):
    return _test_type1(dtype=np.float64, shape=shape, M=M, tol=tol)


def _test_type2(dtype, shape=(16, 16, 16), M=4096, tol=1e-3):
    complex_dtype = utils._complex_dtype(dtype)

    kxyz = utils.gen_nu_pts(M).astype(dtype)
    fk = utils.gen_uniform_data(shape).astype(complex_dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)
    fk_gpu = gpuarray.to_gpu(fk)

    c_gpu = gpuarray.GPUArray(shape=(M,), dtype=complex_dtype)

    plan = cufinufft(2, shape, -1, tol, dtype=dtype)

    plan.set_nu_pts(M, kxyz_gpu[0], kxyz_gpu[1], kxyz_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    plan.destroy()

    c = c_gpu.get()

    ind = M // 2

    c_est = c[ind]
    c_target = utils.direct_type2(fk, kxyz[:, ind])

    type2_rel_err = np.abs(c_target - c_est) / np.abs(c_target)

    print('Type 2 relative error:', type2_rel_err)

    assert type2_rel_err < 0.01


def test_type2_32(shape=(16, 16, 16), M=4096, tol=1e-3):
    return _test_type2(dtype=np.float32, shape=shape, M=M, tol=tol)


def test_type2_64(shape=(16, 16, 16), M=4096, tol=1e-3):
    return _test_type2(dtype=np.float64, shape=shape, M=M, tol=tol)

def test_set_nu_raises_on_dtype():
    dtype = np.float32
    complex_dtype = np.complex64

    M = 4096
    tol = 1e-3
    shape = (16, 16, 16)
    dim = len(shape)

    kxyz = utils.gen_nu_pts(M, dim=dim).astype(dtype)
    c = utils.gen_nonuniform_data(M).astype(complex_dtype)

    kxyz_gpu = gpuarray.to_gpu(kxyz)
    kxyz_gpu_wrong_type = gpuarray.to_gpu(kxyz.astype(np.float64))
    c_gpu = gpuarray.to_gpu(c)
    fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)

    plan = cufinufft(1, shape, 1, tol, dtype=dtype)

    with pytest.raises(TypeError):
        plan.set_nu_pts(M, kxyz_gpu_wrong_type[0],
                        kxyz_gpu[1], kxyz_gpu[2])
    with pytest.raises(TypeError):
        plan.set_nu_pts(M, kxyz_gpu[0],
                        kxyz_gpu_wrong_type[1], kxyz_gpu[2])
    with pytest.raises(TypeError):
        plan.set_nu_pts(M, kxyz_gpu[0],
                        kxyz_gpu[1], kxyz_gpu_wrong_type[2])
    with pytest.raises(TypeError):
        plan.set_nu_pts(M, kxyz_gpu_wrong_type[0],
                        kxyz_gpu_wrong_type[1], kxyz_gpu_wrong_type[2])

    plan.destroy()


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
    # using .real gives us wrong dtype here...
    c_gpu_wrong_dtype = gpuarray.to_gpu(c.real)

    kxyz_gpu = gpuarray.to_gpu(kxyz)
    fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)
    fk_gpu_wrong_dtype = gpuarray.GPUArray(shape, dtype=np.complex128)

    plan = cufinufft(1, shape, 1, tol, dtype=dtype)

    plan.set_nu_pts(M, kxyz_gpu[0],
                    kxyz_gpu[1], kxyz_gpu[2])

    with pytest.raises(TypeError):
        plan.execute(c_gpu, fk_gpu_wrong_dtype)

    with pytest.raises(TypeError):
        plan.execute(c_gpu_wrong_dtype, fk_gpu)

    plan.destroy()


def main():
    test_type1_32()
    test_type2_32()


if __name__ == '__main__':
    main()
