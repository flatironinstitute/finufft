import pytest

import numpy as np

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

from cufinufft import cufinufft

import utils


def test_multi_type1(dtype=np.float32, shape=(16, 16, 16), M=4096, tol=1e-3):
    complex_dtype = utils._complex_dtype(dtype)

    drv.init()

    dev_count = drv.Device.count()

    if dev_count == 1:
        pytest.skip()

    devs = [drv.Device(dev_id) for dev_id in range(dev_count)]

    dim = len(shape)

    errs = []

    for dev_id, dev in enumerate(devs):
        ctx = dev.make_context()

        k = utils.gen_nu_pts(M, dim=dim).astype(dtype)
        c = utils.gen_nonuniform_data(M).astype(complex_dtype)

        k_gpu = gpuarray.to_gpu(k)
        c_gpu = gpuarray.to_gpu(c)
        fk_gpu = gpuarray.GPUArray(shape, dtype=complex_dtype)

        plan = cufinufft(1, shape, eps=tol, dtype=dtype,
                         gpu_device_id=dev_id)

        plan.set_pts(k_gpu[0], k_gpu[1], k_gpu[2])

        plan.execute(c_gpu, fk_gpu)

        fk = fk_gpu.get()

        ind = int(0.1789 * np.prod(shape))

        fk_est = fk.ravel()[ind]
        fk_target = utils.direct_type1(c, k, shape, ind)

        type1_rel_err = np.abs(fk_target - fk_est) / np.abs(fk_target)

        print(f'Type 1 relative error (GPU {dev_id}):', type1_rel_err)

        ctx.pop()

        errs.append(type1_rel_err)

    assert all(err < 0.01 for err in errs)


def main():
    test_multi_type1()


if __name__ == '__main__':
    main()
