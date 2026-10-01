"""Run every script in ``python/finufft/examples``.

The docs embed regions of those scripts, so CI has to execute them or the
embedded text is a claim with no check. Mirrors
``python/cufinufft/tests/test_examples.py``, without the GPU framework filter.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

examples = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))
assert examples, "no example scripts found"


@pytest.mark.parametrize("script", examples, ids=lambda p: p.stem)
def test_example(script):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONOPTIMIZE"}  # keep asserts
    env["OMP_NUM_THREADS"] = str(min(4, os.cpu_count() or 1))
    subprocess.check_call([sys.executable, str(script)], env=env, timeout=600)


def test_impossible_tolerance_fails():
    """Positive control: single precision cannot meet eps=1e-9, so the subprocess must fail with FINUFFT's eps-too-small error."""
    code = (
        "import numpy as np, finufft\n"
        "x = np.random.uniform(-np.pi, np.pi, 1000).astype('float32')\n"
        "c = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype('complex64')\n"
        "finufft.nufft1d1(x, c, 1000, eps=1e-9)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode != 0, "expected subprocess to reject eps=1e-9"
    # must be FINUFFT's eps-too-small failure (ier 26), not an unrelated crash
    assert "RuntimeError: FINUFFT eps tolerance too small to achieve" in proc.stderr, (
        f"unexpected failure mode, stderr: {proc.stderr!r}"
    )
