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


def test_impossible_tolerance_fails(tmp_path):
    """Positive control: single precision cannot meet eps=1e-9, so the subprocess must fail with FINUFFT's eps-too-small error."""
    code = (
        "import numpy as np, finufft\n"
        "x = np.random.uniform(-np.pi, np.pi, 1000).astype('float32')\n"
        "c = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype('complex64')\n"
        "finufft.nufft1d1(x, c, 1000, eps=1e-9)\n"
    )
    # cwd: python -c puts it first on sys.path, so the source tree would shadow the install
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode != 0, "expected subprocess to reject eps=1e-9"
    # must be FINUFFT's eps-too-small failure (ier 26), not an unrelated crash
    assert "RuntimeError: FINUFFT eps tolerance too small to achieve" in proc.stderr, (
        f"unexpected failure mode, stderr: {proc.stderr!r}"
    )


def test_example_gate_fails(tmp_path):
    """Positive control: simple1d1.py with eps=1e-12 in its gate must exit nonzero with FAILED, so the example's accuracy check can fire."""
    src = (examples[0].parent / "simple1d1.py").read_text()
    gate = "eps = 1e-6  # default tolerance"
    assert src.count(gate) == 1
    script = tmp_path / "simple1d1_tight.py"
    script.write_text(src.replace(gate, "eps = 1e-12"))
    # cwd: the source tree must not shadow the installed package
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode != 0, "tightened gate did not fail"
    assert "FAILED" in proc.stdout + proc.stderr, (
        f"unexpected failure mode: {proc.stderr!r}"
    )
