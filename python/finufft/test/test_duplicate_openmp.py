"""Regression tests for issue #942: on macOS, finufft must keep working when
another libomp is in the process, e.g. Homebrew NumPy through OpenBLAS."""

import os
import pathlib
import subprocess
import sys

import pytest

import finufft._finufft

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")

# Homebrew libomp on arm64 and on x86_64. The wheel build installs it.
FOREIGN_LIBOMP = [
    p
    for p in (
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
    )
    if pathlib.Path(p).exists()
]

# Load and start the foreign runtime first, as NumPy does. RTLD_GLOBAL makes
# dyld bind its weak symbols as for a linked dependency; the ctypes default,
# RTLD_LOCAL, hides the bug. Then run finufft with idle gaps so that its
# worker threads suspend: with a second libomp.dylib the suspend code of one
# copy runs on the state of the other, which crashes (OMP Error #179) or hangs.
SCRIPT = """
import ctypes, sys, time
ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL).omp_get_max_threads()
import numpy as np
import finufft
rng = np.random.default_rng(0)
x = rng.uniform(-np.pi, np.pi, 2000)
c = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
ref = np.exp(1j * np.outer(np.arange(-32, 32), x)) @ c
for _ in range(3):
    f = finufft.nufft1d1(x, c, 64, eps=1e-12, nthreads=2)
    err = np.linalg.norm(f - ref) / np.linalg.norm(ref)
    assert err < 1e-10, err
    time.sleep(0.5)
"""


def openmp_exports(path):
    names = subprocess.run(
        ["nm", "-gUj", path], capture_output=True, text=True, check=True
    ).stdout.split()
    return [n for n in names if n.startswith("_omp_") or "_kmp" in n]


def test_no_exported_openmp_symbols():
    assert openmp_exports(finufft._finufft.lib._name) == []
    # The check must find the symbols that a non-hidden runtime exports.
    if FOREIGN_LIBOMP:
        assert openmp_exports(FOREIGN_LIBOMP[0])


def test_no_private_libomp_dylib():
    deps = subprocess.run(
        ["otool", "-L", finufft._finufft.lib._name],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert ".dylibs/libomp" not in deps, deps


@pytest.mark.skipif(not FOREIGN_LIBOMP, reason="no Homebrew libomp found")
def test_second_libomp_in_process():
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT, FOREIGN_LIBOMP[0]],
        # Workers suspend at once, whatever the caller's KMP_BLOCKTIME.
        env={**os.environ, "KMP_BLOCKTIME": "0"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
