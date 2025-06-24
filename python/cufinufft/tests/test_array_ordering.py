import pytest

import numpy as np

from cufinufft import Plan, _compat

import utils


def test_type1_ordering(to_gpu, to_cpu, dtype=np.complex64, shape=(16, 16, 16), M=4096, tol=1e-3):
    k, c = utils.type1_problem(dtype, shape, M)

    k_gpu = to_gpu(k)
    c_gpu = to_gpu(c)

    plan = Plan(1, shape, eps=tol, dtype=dtype)

    plan.setpts(*k_gpu)

    out = np.empty(shape, dtype=dtype, order="F")

    out_gpu = to_gpu(out)

    with pytest.raises(TypeError, match="following requirement: C") as err:
        plan.execute(c_gpu, out=out_gpu)
