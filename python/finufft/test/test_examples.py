"""Run every script in ``python/finufft/examples``.

The docs embed regions of those scripts, so CI runs them to check the
embedded text. Mirrors ``python/cufinufft/tests/test_examples.py``, without
the GPU framework filter.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

examples = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("script", examples, ids=lambda p: p.stem)
def test_example(script):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONOPTIMIZE"}  # keep asserts
    subprocess.check_call([sys.executable, str(script)], env=env)
