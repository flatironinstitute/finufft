#!/usr/bin/env python3
"""
Compare kernel formula errors vs. kernel width, mirroring matlab/test/wsweepkerrcomp.m.

This script uses erralltypedim.py for the core error computations and calls the
FINUFFT Python bindings to measure kernel support via spreadinterponly for each
kernel formula, producing a figure saved under results/.
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import warnings

from erralltypedim import erralltypedim

import finufft


def _count_nonzero_support(du: np.ndarray) -> int:
    """Return the number of nonzero entries in the spreadinterponly output."""
    return int(np.count_nonzero(np.asarray(du) != 0.0))


def main() -> None:
    """Run the kernel-width sweep and save the comparison plot to results/."""
    import argparse

    p = argparse.ArgumentParser(description="Kernel-width sweep")
    p.add_argument("--sigma", type=float, default=2.00, help="upsampling factor (sigma)")
    p.add_argument("--dim", type=int, default=1, choices=[1, 2, 3], help="dimension to test")
    p.add_argument("--M", type=int, default=1000, help="number of nonuniform points (M)")
    p.add_argument("--Ntot", type=int, default=300, help="total number of modes (Ntot)")
    p.add_argument("--ntr", type=int, default=10, help="# transforms to average per tol (ntr)")
    args = p.parse_args()

    prec = "double"
    myrand = np.random.rand
    M = int(args.M)
    dim = int(args.dim)
    Ntot = int(args.Ntot)
    ntr = int(args.ntr)
    isign = +1
    sigma = float(args.sigma)
    tolsperdecade = 8
    tolstep = 10 ** (-1 / tolsperdecade)
    kfnam = ["ES legacy", "ES Beatty", "KB Beatty", "cont-KB Beatty", "cosh-type", "smoothed cont-KB"]
    kfs = list(range(1, len(kfnam) + 1))

    o = {"upsampfac": sigma, "showwarn": False}
    dims = [False, False, False]
    dims[dim - 1] = True
    nkf = len(kfs)

    mintol = 10 * np.finfo(np.float64).eps
    ntols = max(1, math.ceil(math.log(mintol) / math.log(tolstep)))
    ntols = int(ntols)
    tols = tolstep ** np.arange(ntols, dtype=float)

    print(
        f"{dim}D sigma={sigma:.3g}\tprec={prec} M={M} Ntot={Ntot} ntr={ntr} "
        f"ntols={ntols}, kfs: {' '.join(str(k) for k in kfs)}"
    )

    # suppress repeated FINUFFT eps-too-small warnings during sweeps
    warnings.filterwarnings("ignore", message="FINUFFT eps tolerance too small to achieve")

    errs = np.full((nkf, 3, ntols), np.nan, dtype=float)
    ws = np.zeros((nkf, ntols), dtype=int)

    for t, tol in enumerate(tols):
        for i, kf in enumerate(kfs):
            o["spread_kerformula"] = kf
            errvals, info = erralltypedim(
                M, Ntot, ntr, isign, prec, float(tol), o, myrand, dims
            )
            errs[i, :, t] = np.asarray(errvals)[:, dim - 1]

            o_spread = dict(o)
            o_spread["spreadinterponly"] = 1
            du = finufft.nufft1d1(
                np.array([0.0], dtype=float),
                np.array([1.0], dtype=np.complex128),
                n_modes=100,
                eps=tol,
                isign=isign,
                **o_spread,
            )
            ws[i, t] = _count_nonzero_support(du)

    wmax = int(np.max(ws))
    if wmax < 2:
        raise RuntimeError("Spread support never reached w>=2; cannot draw comparison.")

    ekw = np.full((nkf, 3, wmax + 1), np.nan, dtype=float)
    vkw = np.full_like(ekw, np.nan)

    for w in range(2, wmax + 1):
        for i in range(nkf):
            tt = np.flatnonzero(ws[i] == w)
            if tt.size == 0:
                continue
            slice_errs = errs[i, :, tt]
            # slice_errs may be shape (3, n) or (n, 3) or (3,) if n==1.
            if slice_errs.ndim == 1:
                # single sample: assign directly and set zero variance
                ekw[i, :, w] = slice_errs
                vkw[i, :, w] = 0.0
            else:
                # prefer axis=1 if first dim==3, else axis=0
                if slice_errs.shape[0] == 3:
                    mean_vals = np.nanmean(slice_errs, axis=1)
                    var_vals = np.nanvar(slice_errs, axis=1)
                elif slice_errs.shape[1] == 3:
                    mean_vals = np.nanmean(slice_errs, axis=0)
                    var_vals = np.nanvar(slice_errs, axis=0)
                else:
                    # fallback: compute mean along last axis and hope for shape (3,)
                    mean_vals = np.nanmean(slice_errs, axis=1)
                    var_vals = np.nanvar(slice_errs, axis=1)
                ekw[i, :, w] = mean_vals
                vkw[i, :, w] = var_vals

    fig = plt.figure(figsize=(15, 5))
    results_dir = os.path.abspath(os.path.join(os.getcwd(), "results"))
    os.makedirs(results_dir, exist_ok=True)

    w_values = np.arange(2, wmax + 1)
    min_err = np.nanmin(ekw[:, :, 2:])
    max_err = np.nanmax(ekw[:, :, 2:])

    for y in range(3):
        ax = fig.add_subplot(1, 3, y + 1)
        legs = []
        for i in range(nkf):
            means = ekw[i, y, 2:]
            stddev = np.sqrt(vkw[i, y, 2:])
            valid = np.isfinite(means) & np.isfinite(stddev)
            if not np.any(valid):
                continue
            ax.errorbar(
                w_values[valid],
                means[valid],
                yerr=stddev[valid],
                fmt=".-",
                markersize=4,
                linewidth=1,
                elinewidth=0.6,
                capsize=2,
            )
            legs.append(f"kf={kfs[i]}: {kfnam[i]}")
        ax.set_yscale("log")
        ax.set_xlim(2, wmax)
        if np.isfinite(min_err) and np.isfinite(max_err):
            ax.set_ylim(min_err, max_err)
        ax.set_xlabel("w")
        ax.set_ylabel("mean rel err")
        if legs:
            ax.legend(legs)
        ax.set_title(
            f"{dim}D type {y + 1} {prec}, N_tot={Ntot}, σ={sigma}", pad=6
        )
        ax.tick_params(axis="both", which="both", width=0.5, length=4)
        ax.set_facecolor("white")

    plt.tight_layout()
    outfile = os.path.join(
        results_dir, f"wsweepkerrcomp_{dim}D_{prec}_sig{sigma}.png"
    )
    fig.savefig(outfile)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
