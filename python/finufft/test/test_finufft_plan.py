import pytest

import numpy as np

from finufft import Plan

import utils


# NOTE: Only doing single transformations for now.
SHAPES = [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)]
N_PTS = [10, 11]
DTYPES = [np.complex64, np.complex128]
OUTPUT_ARGS = [False, True]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_pts", N_PTS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_finufft1_plan(dtype, shape, n_pts, output_arg):
    pts, coefs = utils.type1_problem(dtype, shape, n_pts)

    plan = Plan(1, shape, dtype=dtype)

    plan.setpts(*pts)

    if not output_arg:
        sig = plan.execute(coefs)
    else:
        sig = np.empty(shape, dtype=dtype)
        plan.execute(coefs, out=sig)

    utils.verify_type1(pts, coefs, shape, sig, 1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_pts", N_PTS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_finufft2_plan(dtype, shape, n_pts, output_arg):
    pts, sig = utils.type2_problem(dtype, shape, n_pts)

    plan = Plan(2, shape, dtype=dtype)

    plan.setpts(*pts)

    if not output_arg:
        coefs = plan.execute(sig)
    else:
        coefs = np.empty(n_pts, dtype=dtype)
        plan.execute(sig, out=coefs)

    utils.verify_type2(pts, sig, coefs, 1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", list(set(len(shape) for shape in SHAPES)))
@pytest.mark.parametrize("n_source_pts", N_PTS)
@pytest.mark.parametrize("n_target_pts", N_PTS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
def test_finufft3_plan(dtype, dim, n_source_pts, n_target_pts, output_arg):
    source_pts, source_coefs, target_pts = utils.type3_problem(dtype,
            dim, n_source_pts, n_target_pts)

    plan = Plan(3, dim, dtype=dtype)

    plan.setpts(*source_pts, *((None,) * (3 - dim)), *target_pts)

    if not output_arg:
        target_coefs = plan.execute(source_coefs)
    else:
        target_coefs = np.empty(n_target_pts, dtype=dtype)
        plan.execute(source_coefs, out=target_coefs)

    utils.verify_type3(source_pts, source_coefs, target_pts, target_coefs, 1e-6)
