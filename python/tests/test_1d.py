# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import finufft
from finufft import interface

__all__ = [
    "test_nufft1d1", "test_nufft1d1", "test_nufft1d3",
]

def test_nufft1d1(seed=42, iflag=1):
    np.random.seed(seed)

    ms = int(1e4)
    n = int(2e4)
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)
    f = finufft.nufft1d1(x, c, ms, eps=tol, iflag=iflag)

    f0 = interface.dirft1d1(x, c, ms, iflag=iflag)
    assert np.all(np.abs((f - f0) / f0) < 1e-6)


def test_nufft1d2(seed=42, iflag=1):
    np.random.seed(seed)

    ms = int(1e4)
    n = int(2e4)
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)
    f = finufft.nufft1d1(x, c, ms, eps=tol, iflag=iflag)

    c = finufft.nufft1d2(x, f, eps=tol, iflag=iflag)
    c0 = interface.dirft1d2(x, f, iflag=iflag)
    assert np.all(np.abs((c - c0) / c0) < 1e-6)


def test_nufft1d3(seed=42, iflag=1):
    np.random.seed(seed)

    ms = int(1e3)
    n = int(2e3)
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)
    s = 0.5 * n * (1.7 + np.random.uniform(-1.0, 1.0, ms))

    f = finufft.nufft1d3(x, c, s, eps=tol, iflag=iflag)
    f0 = interface.dirft1d3(x, c, s, iflag=iflag)
    assert np.all(np.abs((f - f0) / f0) < 1e-6)
