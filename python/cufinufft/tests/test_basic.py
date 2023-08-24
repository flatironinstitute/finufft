import pytest

import numpy as np

from cufinufft import Plan, _compat

import utils

# NOTE: Tests below fail for tolerance 1e-4 (error executing plan).

DTYPES = [np.float32, np.float64]
SHAPES = [(16,), (16, 16), (16, 16, 16)]
MS = [256, 1024, 4096]
TOLS = [1e-2, 1e-3]
OUTPUT_ARGS = [False, True]
CONTIGUOUS = [False, True]

def _transfer_funcs(module_name):
    if module_name == "pycuda":
        import pycuda.autoinit # NOQA:401
        from pycuda.gpuarray import to_gpu
        def to_cpu(obj):
            return obj.get()
    elif module_name == "cupy":
        import cupy
        def to_gpu(obj):
            return cupy.array(obj)
        def to_cpu(obj):
            return obj.get()
    elif module_name == "numba":
        import numba.cuda
        to_gpu = numba.cuda.to_device
        def to_cpu(obj):
            return obj.copy_to_host()
    elif module_name == "torch":
        import torch
        def to_gpu(obj):
            return torch.as_tensor(obj, device=torch.device("cuda"))
        def to_cpu(obj):
            return obj.cpu().numpy()
    else:
        raise TypeError(f"Unsupported framework: {module_name}")

    return to_gpu, to_cpu

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_type1(framework, dtype, shape, M, tol, output_arg):
    to_gpu, to_cpu = _transfer_funcs(framework)

    complex_dtype = utils._complex_dtype(dtype)

    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = to_gpu(k)
    c_gpu = to_gpu(c)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype)

    # Since k_gpu is an array of shape (dim, M), this will expand to
    # plan.setpts(k_gpu[0], ..., k_gpu[dim]), allowing us to handle all
    # dimensions with the same call.
    plan.setpts(*k_gpu)

    if output_arg:
        fk_gpu = _compat.array_empty_like(c_gpu, shape, dtype=complex_dtype)
        plan.execute(c_gpu, out=fk_gpu)
    else:
        fk_gpu = plan.execute(c_gpu)

    fk = to_cpu(fk_gpu)

    utils.verify_type1(k, c, fk, tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
@pytest.mark.parametrize("contiguous", CONTIGUOUS)
def test_type2(framework, dtype, shape, M, tol, output_arg, contiguous):
    if framework == "pycuda" and not contiguous:
        pytest.skip("Pycuda does not support copy to contiguous")

    to_gpu, to_cpu = _transfer_funcs(framework)

    complex_dtype = utils._complex_dtype(dtype)

    k, fk = utils.type2_problem(dtype, shape, M)

    plan = Plan(2, shape, eps=tol, dtype=complex_dtype)

    if not contiguous and len(shape) > 1:
        fk = fk.copy(order="F")

        def _execute(*args, **kwargs):
            with pytest.warns(UserWarning, match="requirement: C. Copying"):
                return plan.execute(*args, **kwargs)
    else:
        def _execute(*args, **kwargs):
            return plan.execute(*args, **kwargs)

    k_gpu = to_gpu(k)
    fk_gpu = to_gpu(fk)

    plan.setpts(*k_gpu)

    if output_arg:
        c_gpu = _compat.array_empty_like(fk_gpu, (M,), dtype=complex_dtype)
        _execute(fk_gpu, out=c_gpu)
    else:
        c_gpu = _execute(fk_gpu)

    c = to_cpu(c_gpu)

    utils.verify_type2(k, fk, c, tol)


def test_opts(shape=(8, 8, 8), M=32, tol=1e-3):
    to_gpu, to_cpu = _transfer_funcs("pycuda")

    dtype = np.float32

    complex_dtype = utils._complex_dtype(dtype)

    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = to_gpu(k)
    c_gpu = to_gpu(c)
    fk_gpu = _compat.array_empty_like(c_gpu, shape, dtype=complex_dtype)

    plan = Plan(1, shape, eps=tol, dtype=complex_dtype, gpu_sort=False,
                     gpu_maxsubprobsize=10)

    plan.setpts(k_gpu[0], k_gpu[1], k_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    fk = to_cpu(fk_gpu)

    utils.verify_type1(k, c, fk, tol)

