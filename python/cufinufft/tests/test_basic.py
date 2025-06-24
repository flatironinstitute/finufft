import pytest

import numpy as np

from cufinufft import Plan, _compat

import utils

# NOTE: Tests below fail for tolerance 1e-4 (error executing plan).

DTYPES = [np.complex64, np.complex128]
SHAPES = [(16,), (16, 16), (16, 16, 16), (19,), (17, 19), (17, 19, 24)]
MS = [256, 1024, 4096]
TOLS = [1e-3, 1e-6]
OUTPUT_ARGS = [False, True]
CONTIGUOUS = [False, True]
MODEORDS = [0, 1]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
@pytest.mark.parametrize("modeord", MODEORDS)
def test_type1(to_gpu, to_cpu, dtype, shape, M, tol, output_arg, modeord):
    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = to_gpu(k)
    c_gpu = to_gpu(c)

    plan = Plan(1, shape, eps=tol, dtype=dtype, modeord=modeord)

    # Since k_gpu is an array of shape (dim, M), this will expand to
    # plan.setpts(k_gpu[0], ..., k_gpu[dim]), allowing us to handle all
    # dimensions with the same call.
    plan.setpts(*k_gpu)

    if output_arg:
        fk_gpu = _compat.array_empty_like(c_gpu, shape, dtype=dtype)
        plan.execute(c_gpu, out=fk_gpu)
    else:
        fk_gpu = plan.execute(c_gpu)

    fk = to_cpu(fk_gpu)
    if modeord == 1:
        fk = np.fft.fftshift(fk)

    utils.verify_type1(k, c, fk, tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
@pytest.mark.parametrize("contiguous", CONTIGUOUS)
@pytest.mark.parametrize("modeord", MODEORDS)
def test_type2(to_gpu, to_cpu, dtype, shape, M, tol, output_arg, contiguous, modeord):
    k, fk = utils.type2_problem(dtype, shape, M)

    plan = Plan(2, shape, eps=tol, dtype=dtype, modeord=modeord)

    check_result = True

    if not contiguous and len(shape) > 1:
        fk = fk.copy(order="F")

        if _compat.array_can_contiguous(to_gpu(np.empty(1))):
            def _execute(*args, **kwargs):
                with pytest.warns(UserWarning, match="requirement: C. Copying"):
                    return plan.execute(*args, **kwargs)
        else:
            check_result = False

            def _execute(*args, **kwargs):
                with pytest.raises(TypeError, match="requirement: C"):
                    plan.execute(*args, **kwargs)

    else:
        def _execute(*args, **kwargs):
            return plan.execute(*args, **kwargs)

    k_gpu = to_gpu(k)

    if modeord == 1:
        _fk = np.fft.ifftshift(fk)
    else:
        _fk = fk
    fk_gpu = to_gpu(_fk)

    plan.setpts(*k_gpu)

    if output_arg:
        c_gpu = _compat.array_empty_like(fk_gpu, (M,), dtype=dtype)
        _execute(fk_gpu, out=c_gpu)
    else:
        c_gpu = _execute(fk_gpu)

    if check_result:
        c = to_cpu(c_gpu)

        utils.verify_type2(k, fk, c, tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", list(set(len(shape) for shape in SHAPES)))
@pytest.mark.parametrize("n_source_pts", MS)
@pytest.mark.parametrize("n_target_pts", MS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_type3(to_gpu, to_cpu, dtype, dim, n_source_pts, n_target_pts, output_arg):
    if dtype == np.float32 and dim >= 2 and min(n_source_pts, n_target_pts) > 4000:
        pytest.xfail("Garbage result for larger numbers of pts in single precision type 3")
        # Strangely, this does not reproduce if we isolate the single case. To
        # trigger it, we must run many other tests preceding this test case.
        # So it's related to some global state of the library.

    source_pts, source_coefs, target_pts = utils.type3_problem(dtype,
            dim, n_source_pts, n_target_pts)

    plan = Plan(3, dim, dtype=dtype)

    source_pts_gpu = to_gpu(source_pts)
    target_pts_gpu = to_gpu(target_pts)

    source_coefs_gpu = to_gpu(source_coefs)

    plan.setpts(*source_pts_gpu, *((None,) * (3 - dim)), *target_pts_gpu)

    if not output_arg:
        target_coefs_gpu = plan.execute(source_coefs_gpu)
    else:
        target_coefs_gpu = _compat.array_empty_like(source_coefs_gpu,
                n_target_pts, dtype=dtype)
        plan.execute(source_coefs_gpu, out=target_coefs_gpu)

    target_coefs = to_cpu(target_coefs_gpu)

    utils.verify_type3(source_pts, source_coefs, target_pts, target_coefs, 1e-6)


def test_opts(to_gpu, to_cpu, shape=(8, 8, 8), M=32, tol=1e-3):
    dtype = np.complex64

    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = to_gpu(k)
    c_gpu = to_gpu(c)
    fk_gpu = _compat.array_empty_like(c_gpu, shape, dtype=dtype)

    plan = Plan(1, shape, eps=tol, dtype=dtype, gpu_sort=False,
                     gpu_maxsubprobsize=10)

    plan.setpts(k_gpu[0], k_gpu[1], k_gpu[2])

    plan.execute(c_gpu, fk_gpu)

    fk = to_cpu(fk_gpu)

    utils.verify_type1(k, c, fk, tol)


def test_cufinufft_plan_properties():
    nufft_type = 2
    n_modes = (8, 8)
    n_trans = 2
    dtype = np.complex64

    plan = Plan(nufft_type, n_modes, n_trans, dtype=dtype)

    assert plan.type == nufft_type
    assert tuple(plan.n_modes) == n_modes
    assert plan.dim == len(n_modes)
    assert plan.n_trans == n_trans
    assert plan.dtype == dtype

    with pytest.raises(AttributeError):
        plan.type = 1

    with pytest.raises(AttributeError):
        plan.n_modes = (4, 4)

    with pytest.raises(AttributeError):
        plan.dim = 1

    with pytest.raises(AttributeError):
        plan.n_trans = 1

    with pytest.raises(AttributeError):
        plan.dtype = np.float64
