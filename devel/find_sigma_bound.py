#!/usr/bin/env python3
"""Validate the sigma_min estimator used in kernel.cpp (finufft::common).

The estimator predicts the minimum upsampling factor (sigma) needed to achieve
a requested tolerance, used by makeplan to warn when upsampfac is too low.

Error model (matches C++ estimated_tol / lowest_sigma in kernel.cpp):

    eps_l2(sigma) = eps_alias + eps_round

    eps_alias = tolfac * exp(-(ns - 1) * pi * u),   u = sqrt(1 - 1/sigma)
    eps_round = 0.48 * eps_mach * N

  tolfac = 0.18 * 1.4^(dim-1), same as theoretical_kernel_ns (kf>=2 branch).
  The aliasing rate pi*u is optimal for ES/KB/PSWF kernels ([BAR] Thm 1).
  The floor coefficient 0.48 is calibrated from [FIN] Remark 9 (N*eps phase error).

sigma_min uses two regimes based on r = tol / eps_round:
  - Kernel regime (r >= 10): pure analytical inversion of the aliasing formula,
    exact to ~0.0001 sigma.
  - Transition regime (r < 10): sigma = sigma_pure + poly(1/r), where the degree-2
    polynomial in 1/r captures the rounding floor effect. Coefficients fit by
    least-squares on empirical data across N=50..5000, types 1-3, dim 1.
    Separate coefficients for double (ns>8) and float (ns<=8).

References:
  [FIN] Barnett, Magland & af Klinteberg, SISC 2019, arxiv:1808.06736
  [BAR] Barnett, Appl. Comput. Harmon. Anal. 2021, arxiv:2001.09405
  Based on Martin Reineck's 3-term PSWF model (PR #841).

Usage (from repo root, after cmake -B build -DFINUFFT_BUILD_TESTS=ON && cmake --build build -j):

    # Generate training data (~30s, parallel over N):
    for N in 50 500 5000; do
      build/devel/find_sigma_bound --prec d,f --type 1,2,3 --N $N --ntol 40 \
        --sigma-prec 1e-3 > /tmp/sigma_N${N}.csv &
    done; wait
    # Combine into one CSV:
    head -1 /tmp/sigma_N50.csv > /tmp/sigma.csv
    for N in 50 500 5000; do tail -n+2 /tmp/sigma_N${N}.csv >> /tmp/sigma.csv; done
    # Plot:
    uv run --with scipy --with matplotlib --with numpy \
      devel/find_sigma_bound.py /tmp/sigma.csv /tmp

Barbone scattered code all around cleaned up by Claude 4.6
"""

import sys

import numpy as np
from scipy.special import pro_ang1

PI = np.pi
MAXSIGMA = 2.0

# Model coefficients — must match kernel.cpp estimated_tol / lowest_sigma.
TOLFAC = 0.18  # kernel aliasing prefactor (same as theoretical_kernel_ns kf>=2)
DIM_FAC = 1.4  # per-dimension correction: tolfac *= 1.4^(dim-1)
FLOOR_C = 0.48  # eps_round = FLOOR_C * eps_mach * N ([FIN] Remark 9)

# Poly(1/r) correction coefficients {a2, a1, a0} for the transition region,
# fit by least-squares across all types, N=50..5000 (see this file's Usage).
POLY_DOUBLE = (0.014, 0.291, -0.043)  # ns > 8
POLY_FLOAT = (0.555, -0.290, 0.071)  # ns <= 8


def _prec_params(prec):
    """Return (ns, eps_mach) for precision char 'd' or 'f'."""
    if prec == "d":
        return 16, 2.2e-16
    return 8, 1.2e-7


# ---------------------------------------------------------------------------
# Model (matches C++ estimated_tol / lowest_sigma in kernel.cpp)
# ---------------------------------------------------------------------------
def _invert_kernel_sigma(tol, ns, dim):
    """Pure analytical sigma inversion (kernel formula only, no floor)."""
    tolfac = TOLFAC * DIM_FAC ** (dim - 1)
    if tol <= 0:
        return MAXSIGMA
    if tol >= tolfac:
        return 1.0 + 1e-6
    u = np.log(tolfac / tol) / ((ns - 1.0) * PI)
    if u >= 1.0:
        return MAXSIGMA
    return min(1.0 / (1.0 - u * u), MAXSIGMA)


def sigma_min_from_model(tol, ns, dim, eps_mach, gridlen):
    """Minimum sigma via hybrid model (matches C++ lowest_sigma)."""
    eps_round = FLOOR_C * eps_mach * gridlen
    r = tol / eps_round
    if r <= 0.5:
        return MAXSIGMA
    sigma_pure = _invert_kernel_sigma(tol, ns, dim)
    if r >= 10.0:
        return sigma_pure
    a2, a1, a0 = POLY_DOUBLE if ns > 8 else POLY_FLOAT
    inv_r = 1.0 / r
    correction = (a2 * inv_r + a1) * inv_r + a0
    return min(sigma_pure + max(correction, 0), MAXSIGMA)


# ---------------------------------------------------------------------------
# mreineck 3-term PSWF model (PR #841, for comparison in plots)
# ---------------------------------------------------------------------------
def _pswf(c, x):
    if abs(x) >= 1.0:
        return float("inf")
    val_0 = pro_ang1(0, 0, c, 0.0)[0]
    if val_0 == 0.0:
        return float("inf")
    return pro_ang1(0, 0, c, x)[0] / val_0


def _mreineck_sigma_min(tol, ns, dim, type_, eps_mach, gridlen):
    """mreineck's 3-term model from PR #841: kernel + rdyn + phase error."""
    tolfac = 0.18 * 1.4 ** (dim - 1)

    def mreineck_tol(sigma):
        # Kernel aliasing
        tol1 = tolfac / np.exp(ns * PI * np.sqrt(1.0 - 1.0 / sigma))
        # Dynamic range of deconvolution correction
        c = PI * ns * (1.0 - 1.0 / (2.0 * sigma)) - 0.05
        pswf_arg = PI * ns / (2.0 * sigma * c)
        if pswf_arg >= 1.0:
            return float("inf")
        pv = _pswf(c, pswf_arg)
        rdyn = 1.0 / pv if 0 < pv < float("inf") else float("inf")
        tol0 = eps_mach * rdyn**dim
        # Phase error from coordinate rounding (N-dependent)
        tol2 = 0.5 * eps_mach * gridlen * sigma
        return tol0 + tol1 + tol2

    if mreineck_tol(MAXSIGMA) > tol:
        return MAXSIGMA + 1.0
    lo, hi = 1.01, MAXSIGMA
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mreineck_tol(mid) <= tol:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_per_N(data, out_prefix):
    """One figure per precision: rows=N, cols=types.  3 lines each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for prec in ("d", "f"):
        ns, eps_mach = _prec_params(prec)
        prec_label = "double" if prec == "d" else "float"
        prec_mask = (data["prec"] == prec) & (data["dim"] == 1)
        d_prec = data[prec_mask]
        if len(d_prec) == 0:
            continue

        Nvals = sorted(set(d_prec["N"].astype(float)))
        types = sorted(set(d_prec["type"].tolist()))
        nrows, ncols = len(Nvals), len(types)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False
        )
        for row, nv in enumerate(Nvals):
            for col, type_ in enumerate(types):
                mask = (
                    prec_mask
                    & (data["type"] == type_)
                    & (data["N"].astype(float) == nv)
                )
                d = data[mask]
                if len(d) == 0:
                    axes[row, col].set_visible(False)
                    continue

                ax = axes[row, col]
                tol = d["tol"].astype(float)
                se = d["sigma_empirical"].astype(float)
                pred = np.array(
                    [sigma_min_from_model(t, ns, 1, eps_mach, nv) for t in tol]
                )
                mr = np.array(
                    [
                        np.clip(
                            _mreineck_sigma_min(t, ns, 1, type_, eps_mach, nv),
                            1.0,
                            2.05,
                        )
                        for t in tol
                    ]
                )

                ax.semilogx(tol, se, "ko-", ms=4, lw=1.2, label="empirical", zorder=5)
                ax.semilogx(
                    tol, np.clip(pred, 1.0, 2.05), "r-", lw=2.5, label="model", zorder=4
                )
                ax.semilogx(
                    tol, mr, color="#1f77b4", ls="--", lw=2, label="mreineck", zorder=3
                )
                ax.set_xlabel("tolerance")
                ax.set_ylabel(r"$\sigma_{\min}$")
                ax.set_title(f"{prec_label} type {type_}  N={int(nv)}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(1.0 if prec == "f" else 1.3, 2.15)

        fig.suptitle(
            f"{prec_label} (ns={ns}): empirical vs model vs mreineck",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        out = f"{out_prefix}_{prec_label}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv> [output_dir]", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
    data = np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )

    # Error summary
    print("max |sigma_predicted - sigma_empirical|:")
    combos = sorted(
        set(
            zip(
                data["prec"].tolist(),
                data["type"].tolist(),
                data["dim"].tolist(),
            )
        )
    )
    for prec, type_, dim in combos:
        ns, eps_mach = _prec_params(prec)
        mask = (data["prec"] == prec) & (data["type"] == type_) & (data["dim"] == dim)
        d = data[mask]
        if len(d) == 0:
            continue
        tol = d["tol"].astype(float)
        se = d["sigma_empirical"].astype(float)
        N_arr = d["N"].astype(float)
        pred = np.array(
            [
                sigma_min_from_model(t, ns, int(dim), eps_mach, float(N_arr[i]))
                for i, t in enumerate(tol)
            ]
        )
        valid = pred <= 2.05
        diffs = pred[valid] - se[valid]
        mu = max(-diffs.min(), 0) if len(diffs) > 0 else 0
        mo = max(diffs.max(), 0) if len(diffs) > 0 else 0
        label = "double" if prec == "d" else "float "
        print(f"  {label} type={type_} dim={dim}: under={mu:.3f} over={mo:.3f}")

    # Plots
    print("\nPlots:")
    plot_per_N(data, f"{out_dir}/sigma_fit")


if __name__ == "__main__":
    main()
