"""Run every script in ``python/finufft/examples``.

The docs embed regions of those scripts, so CI has to execute them or the
embedded text is a claim with no check. Mirrors
``python/cufinufft/tests/test_examples.py``, without the GPU framework filter.
"""

import subprocess
import sys
from pathlib import Path

import pytest

examples = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("script", examples, ids=lambda p: p.stem)
def test_example(script):
    subprocess.check_call([sys.executable, str(script)])


def test_impossible_tolerance_fails():
    """Positive control: single precision cannot meet eps=1e-9, so the subprocess must exit nonzero."""
    code = (
        "import numpy as np, finufft\n"
        "x = np.random.uniform(-np.pi, np.pi, 1000).astype('float32')\n"
        "c = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype('complex64')\n"
        "finufft.nufft1d1(x, c, 1000, eps=1e-9)\n"
    )
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call([sys.executable, "-c", code])
    print(
        "positive control: eps=1e-9 in single precision correctly failed the subprocess"
    )
