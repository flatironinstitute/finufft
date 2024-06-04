import pytest

import numpy as np

from finufft import Plan

import utils


# NOTE: Only doing single transformations for now.
SHAPES = [(7,), (8,), (7, 7), (7, 8), (8, 8), (7, 7, 7), (7, 8, 8), (8, 8, 8)]
N_PTS = [10, 11]
DTYPES = [np.complex64, np.complex128]
OUTPUT_ARGS = [False, True]
MODEORDS = [0, 1]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_pts", N_PTS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
@pytest.mark.parametrize("modeord", MODEORDS)
def test_finufft1_plan(dtype, shape, n_pts, output_arg, modeord):
    pts, coefs = utils.type1_problem(dtype, shape, n_pts)

    plan = Plan(1, shape, dtype=dtype, modeord=modeord)

    plan.setpts(*pts)

    if not output_arg:
        sig = plan.execute(coefs)
    else:
        sig = np.empty(shape, dtype=dtype)
        plan.execute(coefs, out=sig)

    if modeord == 1:
        sig = np.fft.fftshift(sig)

    utils.verify_type1(pts, coefs, shape, sig, 1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_pts", N_PTS)
@pytest.mark.parametrize("output_arg", OUTPUT_ARGS)
@pytest.mark.parametrize("modeord", MODEORDS)
def test_finufft2_plan(dtype, shape, n_pts, output_arg, modeord):
    pts, sig = utils.type2_problem(dtype, shape, n_pts)

    plan = Plan(2, shape, dtype=dtype, modeord=modeord)

    plan.setpts(*pts)

    if modeord == 1:
        _sig = np.fft.ifftshift(sig)
    else:
        _sig = sig

    if not output_arg:
        coefs = plan.execute(_sig)
    else:
        coefs = np.empty(n_pts, dtype=dtype)
        plan.execute(_sig, out=coefs)

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


def test_finufft_plan_errors():
    with pytest.raises(RuntimeError, match="must be single or double"):
        Plan(1, (8, 8), dtype="uint32")

    with pytest.warns(Warning, match="finufft_opts does not have"):
        Plan(1, (8, 8), foo="bar")

    with pytest.raises(RuntimeError, match="type 3 plan n_modes_or_dim must be"):
        Plan(3, (1, 2))

    with pytest.raises(RuntimeError, match="n_modes dimension must be 1, 2"):
        Plan(2, (1, 2, 3, 4))

    with pytest.warns(Warning, match="eps tolerance too small"):
        Plan(1, (8, 8), eps=1e-30)

    with pytest.raises(TypeError, match="does not have the correct dtype"):
        Plan(1, (8, 8), dtype="complex64").setpts(np.ones(1, dtype="complex128"))

    with pytest.raises(TypeError, match="the following requirement: C"):
        plan = Plan(1, (8, 8), dtype="complex64")
        plan.setpts(*np.ones((2, 1), dtype="float32"))
        plan.execute(np.ones(1, dtype="complex64"), out=np.ones((8, 8),
            dtype="complex64", order="F"))

    with pytest.raises(TypeError, match="the following requirement: W"):
        plan = Plan(1, (8, 8), dtype="complex64")
        plan.setpts(*np.ones((2, 1), dtype="float32"))
        out = np.ones((8, 8), dtype="complex64")
        out.setflags(write=False)
        plan.execute(np.ones(1, dtype="complex64"), out=out)

    with pytest.warns(Warning, match="the following requirement: C. Copying"):
        plan = Plan(2, (8, 8), dtype="complex64")
        plan.setpts(*np.ones((2, 1), dtype="float32"))
        plan.execute(np.ones((8, 8), dtype="complex64", order="F"))

    vec = np.ones(1, dtype="float32")
    not_vec = np.ones((2, 1), dtype="float32")

    with pytest.raises(RuntimeError, match="x must be a vector"):
        Plan(1, (8, 8, 8), dtype="complex64").setpts(not_vec, vec, vec)

    with pytest.raises(RuntimeError, match="y must be a vector"):
        Plan(1, (8, 8, 8), dtype="complex64").setpts(vec, not_vec, vec)

    with pytest.raises(RuntimeError, match="z must be a vector"):
        Plan(1, (8, 8, 8), dtype="complex64").setpts(vec, vec, not_vec)

    with pytest.raises(RuntimeError, match="s must be a vector"):
        Plan(3, 3, dtype="complex64").setpts(vec, vec, vec, not_vec, vec, vec)

    with pytest.raises(RuntimeError, match="t must be a vector"):
        Plan(3, 3, dtype="complex64").setpts(vec, vec, vec, vec, not_vec, vec)

    with pytest.raises(RuntimeError, match="u must be a vector"):
        Plan(3, 3, dtype="complex64").setpts(vec, vec, vec, vec, vec, not_vec)

    vec = np.ones(3, dtype="float32")
    long_vec = np.ones(4, dtype="float32")

    with pytest.raises(RuntimeError, match="y must have same length as x"):
        Plan(1, (8, 8, 8), dtype="complex64").setpts(vec, long_vec, vec)

    with pytest.raises(RuntimeError, match="z must have same length as x"):
        Plan(1, (8, 8, 8), dtype="complex64").setpts(vec, vec, long_vec)

    with pytest.raises(RuntimeError, match="t must have same length as s"):
        Plan(3, 3, dtype="complex64").setpts(vec, vec, vec, vec, long_vec, vec)

    with pytest.raises(RuntimeError, match="u must have same length as s"):
        Plan(3, 3, dtype="complex64").setpts(vec, vec, vec, vec, vec, long_vec)

    with pytest.raises(RuntimeError, match="c.ndim must be 1 or 2 if n_trans == 1"):
        plan = Plan(1, (8,), dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"))
        plan.execute(np.ones((2, 1, 1), dtype="complex64"))

    with pytest.raises(RuntimeError, match="c.size must be same as x.size"):
        plan = Plan(1, (8,), dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"))
        plan.execute(np.ones(4, dtype="complex64"))

    with pytest.raises(RuntimeError, match="c.ndim must be 2 if n_trans > 1"):
        plan = Plan(1, (8,), n_trans=2, dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"))
        plan.execute(np.ones(2, dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"c.shape must be \(n_trans, x.size\)"):
        plan = Plan(1, (8,), n_trans=2, dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"))
        plan.execute(np.ones((2, 4), dtype="complex64"))

    with pytest.raises(RuntimeError, match="same as the problem dimension"):
        plan = Plan(2, (8,), dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"))
        plan.execute(np.ones((1, 2, 8), dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"same as the problem dimension \+ 1 for"):
        plan = Plan(2, (8,), n_trans=2, dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"))
        plan.execute(np.ones(8, dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"f\.shape\[0\] must be n_trans"):
        plan = Plan(2, (8,), n_trans=2, dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"))
        plan.execute(np.ones((3, 8), dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"f\.shape is not consistent"):
        plan = Plan(2, (8,), dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"))
        plan.execute(np.ones(2, dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"f\.shape is not consistent"):
        plan = Plan(2, (8, 9), dtype="complex64")
        plan.setpts(*np.ones((2, 3), dtype="float32"))
        plan.execute(np.ones((2, 9), dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"f\.shape is not consistent"):
        plan = Plan(2, (8, 9, 10), dtype="complex64")
        plan.setpts(*np.ones((3, 3), dtype="float32"))
        plan.execute(np.ones((2, 9, 10), dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"f\.ndim must be 1 or 2"):
        plan = Plan(3, 1, dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"), s=np.ones(3, dtype="float32"))
        plan.execute(np.ones(3, dtype="complex64"), out=np.ones((1, 2, 3), dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"f\.size of must be nk"):
        plan = Plan(3, 1, dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"), s=np.ones(3, dtype="float32"))
        plan.execute(np.ones(3, dtype="complex64"), out=np.ones(4, dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"f\.ndim must be 2"):
        plan = Plan(3, 1, n_trans=2, dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"), s=np.ones(3, dtype="float32"))
        plan.execute(np.ones((2, 3), dtype="complex64"), out=np.ones(3, dtype="complex64"))

    with pytest.raises(RuntimeError, match=r"f\.shape must be \(n_trans, nk\)"):
        plan = Plan(3, 1, n_trans=2, dtype="complex64")
        plan.setpts(np.ones(3, dtype="float32"), s=np.ones(3, dtype="float32"))
        plan.execute(np.ones((2, 3), dtype="complex64"), out=np.ones((2, 4), dtype="complex64"))

    with pytest.raises(RuntimeError, match="transform type invalid"):
        plan = Plan(4, (8,))
