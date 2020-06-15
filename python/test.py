import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from cufinufft import cufinufft

import utils


def test_type1(shape=(16, 16, 16), M=4096, tol=1e-3):
    kxyz = utils.gen_nu_pts(M)
    c = utils.gen_nonuniform_data(M)

    kxyz_gpu = gpuarray.to_gpu(kxyz)
    c_gpu = gpuarray.to_gpu(c)
    fk_gpu = gpuarray.GPUArray(shape, dtype=np.complex64)

    plan = cufinufft(1, shape, 1, tol)

    plan.set_nu_pts(M, kxyz_gpu[0], kxyz_gpu[1], kxyz_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    fk = fk_gpu.get()

    ind = int(0.1789 * np.prod(shape))

    fk_est = fk.ravel()[ind]
    fk_target = utils.direct_type1(c, kxyz, shape, ind)

    type1_rel_err = np.abs(fk_target - fk_est) / np.abs(fk_target)

    print('Type 1 relative error:', type1_rel_err)

    plan.destroy()

def test_type2(shape=(16, 16, 16), M=4096, tol=1e-3):
    kxyz = utils.gen_nu_pts(M)
    fk = utils.gen_uniform_data(shape)

    kxyz_gpu = gpuarray.to_gpu(kxyz)
    fk_gpu = gpuarray.to_gpu(fk)

    c_gpu = gpuarray.GPUArray(shape=(M,), dtype=np.complex64)

    plan = cufinufft(2, shape, -1, tol)

    plan.set_nu_pts(M, kxyz_gpu[0], kxyz_gpu[1], kxyz_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    plan.destroy()

    c = c_gpu.get()

    ind = M // 2

    c_est = c[ind]
    c_target = utils.direct_type2(fk, kxyz[:, ind])

    type2_rel_err = np.abs(c_target - c_est) / np.abs(c_target)

    print('Type 2 relative error:', type2_rel_err)


def main():
    test_type1()
    test_type2()


if __name__ == '__main__':
    main()
