import numpy as np

import pycuda.autoinit # NOQA:401
import pycuda.gpuarray as gpuarray

from cufinufft import cufinufft

import utils


def _test_type1(dtype, shape=(16, 16, 16), M=4096, tol=1e-3):
    complex_dtype = utils._complex_dtype(dtype)

    dim = len(shape)

    k = utils.gen_nu_pts(M, dim=dim).astype(dtype)
    c = utils.gen_nonuniform_data(M).astype(complex_dtype)

    k_gpu = gpuarray.to_gpu(k)
    c_gpu = gpuarray.to_gpu(c)
    fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)

    plan = cufinufft(1, shape, eps=tol, dtype=dtype)

    plan.set_pts(k_gpu[0], k_gpu[1], k_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    fk = fk_gpu.get()

    ind = int(0.1789 * np.prod(shape))

    fk_est = fk.ravel()[ind]
    fk_target = utils.direct_type1(c, k, shape, ind)

    type1_rel_err = np.abs(fk_target - fk_est) / np.abs(fk_target)

    print('Type 1 relative error:', type1_rel_err)

    assert type1_rel_err < 0.01


def test_type1_32(shape=(16, 16, 16), M=4096, tol=1e-3):
    return _test_type1(dtype=np.float32, shape=shape, M=M, tol=tol)


def test_type1_64(shape=(16, 16, 16), M=4096, tol=1e-3):
    return _test_type1(dtype=np.float64, shape=shape, M=M, tol=tol)


def _test_type2(dtype, shape=(16, 16, 16), M=4096, tol=1e-3):
    complex_dtype = utils._complex_dtype(dtype)

    k = utils.gen_nu_pts(M).astype(dtype)
    fk = utils.gen_uniform_data(shape).astype(complex_dtype)

    k_gpu = gpuarray.to_gpu(k)
    fk_gpu = gpuarray.to_gpu(fk)

    c_gpu = gpuarray.GPUArray(shape=(M,), dtype=complex_dtype)

    plan = cufinufft(2, shape, eps=tol, dtype=dtype)

    plan.set_pts(k_gpu[0], k_gpu[1], k_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    c = c_gpu.get()

    ind = M // 2

    c_est = c[ind]
    c_target = utils.direct_type2(fk, k[:, ind])

    type2_rel_err = np.abs(c_target - c_est) / np.abs(c_target)

    print('Type 2 relative error:', type2_rel_err)

    assert type2_rel_err < 0.01


def test_type2_32(shape=(16, 16, 16), M=4096, tol=1e-3):
    return _test_type2(dtype=np.float32, shape=shape, M=M, tol=tol)


def test_type2_64(shape=(16, 16, 16), M=4096, tol=1e-3):
    return _test_type2(dtype=np.float64, shape=shape, M=M, tol=tol)


def test_opts(shape=(8, 8, 8), M=32, tol=1e-3):
    dtype = np.float32

    complex_dtype = utils._complex_dtype(dtype)

    dim = len(shape)

    k = utils.gen_nu_pts(M, dim=dim).astype(dtype)
    c = utils.gen_nonuniform_data(M).astype(complex_dtype)

    k_gpu = gpuarray.to_gpu(k)
    c_gpu = gpuarray.to_gpu(c)
    fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)

    plan = cufinufft(1, shape, eps=tol, dtype=dtype, gpu_sort=False,
                     gpu_maxsubprobsize=10)

    plan.set_pts(k_gpu[0], k_gpu[1], k_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    fk = fk_gpu.get()

    ind = int(0.1789 * np.prod(shape))

    fk_est = fk.ravel()[ind]
    fk_target = utils.direct_type1(c, k, shape, ind)

    type1_rel_err = np.abs(fk_target - fk_est) / np.abs(fk_target)

    assert type1_rel_err < 0.01


def main():
    test_type1_32()
    test_type2_32()
    test_type1_64()
    test_type2_64()


if __name__ == '__main__':
    main()
