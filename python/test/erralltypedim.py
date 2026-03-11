#!/usr/bin/env python3
"""
erralltypedim using real FINUFFT Python bindings.

Matches MATLAB logic and returns:
  err : (3,3) numpy array of relative 2-norm errors
  info: dict with at least key 'Nmax' -> list of 3 ints

Signature:
  err, info = erralltypedim(M, Ntot, ntr, isign, prec, tol, o, myrand, dims, errcheck=-1)
"""
from __future__ import annotations
import math
from typing import Any, Dict, Tuple
import os, sys
import numpy as np

import finufft

def _rand_call(myrand, *shape):
    try:
        return myrand(*shape)
    except TypeError:
        return np.asarray(myrand(*shape))

def erralltypedim(M: int, Ntot: int, ntr: int, isign: int, prec: str,
                  tol: float, o: Dict[str, Any], myrand, dims, errcheck: float = -1
                  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    err = np.full((3,3), np.nan, dtype=float)
    info: Dict[str, Any] = {"Nmax": [np.nan, np.nan, np.nan]}

    # select dtypes
    if prec.startswith("s") or prec == "single":
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # generate random pts in [0,1) then scale to [0,2pi)
    xr = np.ascontiguousarray(_rand_call(myrand, M, 1).reshape(M), dtype=real_dtype)
    yr = np.ascontiguousarray(_rand_call(myrand, M, 1).reshape(M), dtype=real_dtype)
    zr = np.ascontiguousarray(_rand_call(myrand, M, 1).reshape(M), dtype=real_dtype)
    x = np.ascontiguousarray(2.0 * math.pi * xr, dtype=real_dtype)
    y = np.ascontiguousarray(2.0 * math.pi * yr, dtype=real_dtype)
    z = np.ascontiguousarray(2.0 * math.pi * zr, dtype=real_dtype)

    # complex strengths: shape (ntr, M) to match finufft python interface
    cr = np.asarray(_rand_call(myrand, M, ntr)).reshape(M, ntr).T
    cr2 = np.asarray(_rand_call(myrand, M, ntr)).reshape(M, ntr).T
    # ensure C-contiguous (ntr, M)
    real_part = np.ascontiguousarray((2.0 * cr - 1.0), dtype=real_dtype)
    imag_part = np.ascontiguousarray((2.0 * cr2 - 1.0), dtype=real_dtype)
    c = np.ascontiguousarray(real_part + 1j * imag_part, dtype=complex_dtype)

    # 1D
    if dims is None or bool(dims[0]):
        N = int(Ntot)
        info["Nmax"][0] = N
        k = np.arange(math.ceil(-N/2), math.floor((N-1)/2) + 1, dtype=int)
        A = np.exp(1j * isign * np.outer(k, x))               # (N, M)
        # exact direct: fe (N x ntr) = A @ c.T
        fe = A.dot(c.T)                                       # (N, ntr)
        # call FINUFFT: finufft.nufft1d1 expects c shape (ntr, M) and returns (ntr, N)
        # ensure inputs are contiguous and correct dtype
        f = finufft.nufft1d1(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(c, dtype=complex_dtype), n_modes=N, eps=tol, isign=isign, **(o or {}))
        # ensure shape (ntr, N) and C-contiguous
        f = np.ascontiguousarray(np.asarray(f), dtype=complex_dtype)
        # make fe in same shape (ntr, N)
        fe_T = fe.T.astype(complex_dtype)
        err[0,0] = np.linalg.norm(f.ravel() - fe_T.ravel()) / np.linalg.norm(fe_T.ravel())

        # type2: input f (ntr, N) -> output c (ntr, M)
        C = finufft.nufft1d2(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(f, dtype=complex_dtype), eps=tol, isign=isign, **(o or {}))
        C = np.ascontiguousarray(np.asarray(C), dtype=complex_dtype)             # (ntr, M)
        Ce = (A.T.dot(fe)).T.astype(complex_dtype)          # A.' * f  -> (ntr, M)
        err[1,0] = np.linalg.norm(C - Ce) / np.linalg.norm(Ce)

        # type3: choose target freqs s of length M
        s = (N * _rand_call(myrand, M, 1).reshape(M,)).astype(real_dtype)
        f3 = finufft.nufft1d3(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(c, dtype=complex_dtype), np.ascontiguousarray(s, dtype=real_dtype), eps=tol, isign=isign, **(o or {}))
        f3 = np.ascontiguousarray(np.asarray(f3), dtype=complex_dtype)
        fe3 = (np.exp(1j * isign * np.outer(s, x)).dot(c.T)).T.astype(complex_dtype)  # (ntr, M)
        err[2,0] = np.linalg.norm(f3 - fe3) / np.linalg.norm(fe3)

    # 2D
    if dims is None or bool(dims[1]):
        N1 = int(round(math.sqrt(2 * Ntot)))
        N2 = int(round(Ntot / N1)) if N1 != 0 else 1
        info["Nmax"][1] = max(N1, N2)
        kx = np.arange(math.ceil(-N1/2), math.floor((N1-1)/2) + 1, dtype=int)
        ky = np.arange(math.ceil(-N2/2), math.floor((N2-1)/2) + 1, dtype=int)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        kxv = KX.ravel(); kyv = KY.ravel()
        A2 = np.exp(1j * isign * (np.outer(kxv, x) + np.outer(kyv, y)))  # (N1*N2, M)
        f2 = finufft.nufft2d1(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(y, dtype=real_dtype), np.ascontiguousarray(c, dtype=complex_dtype), n_modes=(N1, N2), eps=tol, isign=isign, **(o or {}))
        f2 = np.ascontiguousarray(np.asarray(f2), dtype=complex_dtype)           # (ntr, N1*N2)
        fe2 = A2.dot(c.T).T.astype(complex_dtype)           # (ntr, N1*N2)
        err[0,1] = np.linalg.norm(f2.ravel() - fe2.ravel()) / np.linalg.norm(fe2.ravel())

        C2 = finufft.nufft2d2(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(y, dtype=real_dtype), np.ascontiguousarray(f2, dtype=complex_dtype), eps=tol, isign=isign, **(o or {}))
        C2 = np.ascontiguousarray(np.asarray(C2), dtype=complex_dtype)           # (ntr, M)
        Ce2 = (A2.T.dot(f2.T)).T.astype(complex_dtype)
        err[1,1] = np.linalg.norm(C2 - Ce2) / np.linalg.norm(Ce2)

        s = (N1 * _rand_call(myrand, M, 1).reshape(M,)).astype(real_dtype)
        t = (N2 * _rand_call(myrand, M, 1).reshape(M,)).astype(real_dtype)
        f23 = finufft.nufft2d3(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(y, dtype=real_dtype), np.ascontiguousarray(c, dtype=complex_dtype), np.ascontiguousarray(s, dtype=real_dtype), np.ascontiguousarray(t, dtype=real_dtype), eps=tol, isign=isign, **(o or {}))
        f23 = np.ascontiguousarray(np.asarray(f23), dtype=complex_dtype)
        fe23 = (np.exp(1j * isign * (np.outer(s, x) + np.outer(t, y))).dot(c.T)).T.astype(complex_dtype)
        err[2,1] = np.linalg.norm(f23 - fe23) / np.linalg.norm(fe23)

    # 3D
    if dims is None or bool(dims[2]):
        N1 = int(round((2 * Ntot) ** (1/3)))
        N2 = int(round(Ntot ** (1/3)))
        N3 = int(round(Ntot / N1 / N2)) if (N1 * N2) != 0 else 1
        info["Nmax"][2] = max(N1, N2, N3)
        kx = np.arange(math.ceil(-N1/2), math.floor((N1-1)/2) + 1, dtype=int)
        ky = np.arange(math.ceil(-N2/2), math.floor((N2-1)/2) + 1, dtype=int)
        kz = np.arange(math.ceil(-N3/2), math.floor((N3-1)/2) + 1, dtype=int)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        kxv = KX.ravel(); kyv = KY.ravel(); kzv = KZ.ravel()
        A3 = np.exp(1j * isign * (np.outer(kxv, x) + np.outer(kyv, y) + np.outer(kzv, z)))  # (N1*N2*N3, M)
        f3d = finufft.nufft3d1(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(y, dtype=real_dtype), np.ascontiguousarray(z, dtype=real_dtype), np.ascontiguousarray(c, dtype=complex_dtype), n_modes=(N1, N2, N3), eps=tol, isign=isign, **(o or {}))
        f3d = np.ascontiguousarray(np.asarray(f3d), dtype=complex_dtype)
        fe3d = A3.dot(c.T).T.astype(complex_dtype)
        err[0,2] = np.linalg.norm(f3d.ravel() - fe3d.ravel()) / np.linalg.norm(fe3d.ravel())

        C3 = finufft.nufft3d2(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(y, dtype=real_dtype), np.ascontiguousarray(z, dtype=real_dtype), np.ascontiguousarray(f3d, dtype=complex_dtype), eps=tol, isign=isign, **(o or {}))
        C3 = np.ascontiguousarray(np.asarray(C3), dtype=complex_dtype)
        Ce3 = (A3.T.dot(f3d.T)).T.astype(complex_dtype)
        err[1,2] = np.linalg.norm(C3 - Ce3) / np.linalg.norm(Ce3)

        s = (N1 * _rand_call(myrand, M, 1).reshape(M,)).astype(real_dtype)
        t = (N2 * _rand_call(myrand, M, 1).reshape(M,)).astype(real_dtype)
        u = (N3 * _rand_call(myrand, M, 1).reshape(M,)).astype(real_dtype)
        f33 = finufft.nufft3d3(np.ascontiguousarray(x, dtype=real_dtype), np.ascontiguousarray(y, dtype=real_dtype), np.ascontiguousarray(z, dtype=real_dtype), np.ascontiguousarray(c, dtype=complex_dtype), np.ascontiguousarray(s, dtype=real_dtype), np.ascontiguousarray(t, dtype=real_dtype), np.ascontiguousarray(u, dtype=real_dtype), eps=tol, isign=isign, **(o or {}))
        f33 = np.ascontiguousarray(np.asarray(f33), dtype=complex_dtype)
        fe33 = (np.exp(1j * isign * (np.outer(s, x) + np.outer(t, y) + np.outer(u, z))).dot(c.T)).T.astype(complex_dtype)
        err[2,2] = np.linalg.norm(f33 - fe33) / np.linalg.norm(fe33)

    return err, info

if __name__ == "__main__":
    e, info = erralltypedim(100, 300, 1, +1, "double", 1e-6, {}, np.random.rand, np.array([True, False, False]))
    print("err shape:", e.shape, "Nmax:", info["Nmax"])
