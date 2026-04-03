#!/usr/bin/env python3
# Run with: uv run --with scipy --with matplotlib --with numpy devel/find_sigma_bound.py
"""Derive and validate the sigma_min estimator used in makeplan.hpp.

The estimator predicts the minimum upsampling factor (sigma) needed to achieve a
requested tolerance. It is used to emit warnings when the user's upsampfac is too low.

## Error model

The NUFFT achievable tolerance at a given sigma is modelled as two terms:

    tol(sigma) = tol_kernel + tol_floor

  tol_kernel = tolfac * exp(-(ns - nsoff) * pi * u^u_exp)
      where u = sqrt(1 - 1/sigma)
  tol_floor  = floor_c * eps_mach

The kernel truncation term captures the dominant exponential decay of the spreading
kernel error. The floor term captures the combined contributions of:
  - FFT rounding: O(log2(N) * eps_mach)
  - Deconvolution dynamic range: eps_mach * rdyn (averages out over modes)
  - Horner polynomial approximation error

For production use (makeplan.hpp), a phase term is added:
  tol_phase = 0.5 * eps_mach * gridlen * sigma

## Calibrated coefficients (double, dim=1, ns=16)

Fitted against empirical data from find_sigma_bound.cpp (N=300, ntol=60):

  tolfac_base = 0.0798
  nsoff       = 0.0901  (≈ use full W, not W-1)
  u_exp       = 1.1893  (slightly superlinear — controls descent slope)
  floor_c     = 125.5   (≈ 126 * eps_mach ≈ 2.8e-14 for double)
  dim_factor  = 1.4^(dim-1)  (empirical worsening per dimension)
  type3_factor = 1.1    (type 3 has slightly larger errors)

  tolfac = tolfac_base * dim_factor * (type3_factor if type==3 else 1.0)

Accuracy against training data:
  type 1: max|error| = 0.034 sigma
  type 2: max|error| = 0.028 sigma
  type 3: max|error| = 0.022 sigma

## Derivation history

Starting point: mreineck's 3-term model (flatironinstitute/finufft PR #841):
  tol = tolfac*exp(-(W-1)*pi*u) + eps_mach*rdyn^dim + 0.5*eps_mach*gridlen*sigma

where rdyn = 1/pswf(c, pi*W/(2*sigma*c)), the deconvolution dynamic range.

Key findings during calibration:
1. The rdyn term (worst-case Nyquist amplification) overestimates the actual L2 error
   because most output modes see much lower amplification. Replacing it with a constant
   floor matches the empirical data better.
2. The standard exponent u = sqrt(1-1/sigma) gives a slope that's too steep compared to
   empirical. Raising u to power ~1.19 flattens the descent to match.
3. tolfac = 0.18 (mreineck's value) underestimates the error. The calibrated value
   0.0798 with nsoff≈0 (using full W instead of W-1) gives the right magnitude.
4. The type 3 factor of 1.4 (from FINUFFT's kernel.cpp) is slightly too conservative;
   1.1 matches the empirical data better.

Usage:
  # Generate training data (C++ tool, ~10s):
  cd build && LD_LIBRARY_PATH=src \\
    devel/find_sigma_bound --prec d --type 1,2,3 --dim 1 --ntol 60 --sigma-prec 5e-4 \\
    > /tmp/sigma.csv

  # Validate and plot:
  uv run --with scipy --with matplotlib --with numpy devel/find_sigma_bound.py /tmp/sigma.csv

  # With coefficient calibration:
  uv run --with scipy --with matplotlib --with numpy devel/find_sigma_bound.py --calibrate /tmp/sigma.csv
"""

import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import pro_ang1

PI = np.pi
MAXSIGMA = 2.0

# ---------------------------------------------------------------------------
# Calibrated model coefficients
# ---------------------------------------------------------------------------
TOLFAC_BASE = 0.079
NSOFF = 0.1
U_EXP = 1.19
FLOOR_C = 117.0
DIM_FACTOR = 1.4  # per extra dimension
TYPE3_FACTOR = 1.0  # no special type 3 correction needed


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
def estimated_tol(
    sigma,
    ns,
    dim,
    type_,
    eps_mach,
    tf=TOLFAC_BASE,
    nsoff=NSOFF,
    u_exp=U_EXP,
    floor_c=FLOOR_C,
    t3_fac=TYPE3_FACTOR,
):
    """Estimated achievable tolerance at given sigma."""
    tolfac = tf * DIM_FACTOR ** (dim - 1) * (t3_fac if type_ == 3 else 1.0)
    u = np.sqrt(1.0 - 1.0 / sigma)
    tol_kernel = tolfac * np.exp(-(ns - nsoff) * PI * u**u_exp)
    tol_floor = floor_c * eps_mach
    return tol_kernel + tol_floor


def sigma_min_from_model(tol, ns, dim, type_, eps_mach, **kw):
    """Binary search to find minimum sigma achieving tol."""
    if estimated_tol(MAXSIGMA, ns, dim, type_, eps_mach, **kw) > tol:
        return MAXSIGMA + 1.0
    lo, hi = 1.01, MAXSIGMA
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if estimated_tol(mid, ns, dim, type_, eps_mach, **kw) <= tol:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(data, types=(1, 2, 3), dim=1, prec="d"):
    """Optimize model coefficients against empirical data."""
    ns = 16 if prec == "d" else 8
    eps_mach = 2.2e-16 if prec == "d" else 1.2e-7

    subsets = {}
    for type_ in types:
        mask = (data["prec"] == prec) & (data["type"] == type_) & (data["dim"] == dim)
        d = data[mask]
        if len(d) == 0:
            continue
        subsets[type_] = (d["tol"].astype(float), d["sigma_empirical"].astype(float))

    def objective(params):
        tf, nsoff, u_exp, floor_c, t3_fac = params
        if (
            tf < 0.01
            or nsoff < 0
            or nsoff > 2
            or u_exp < 0.2
            or floor_c < 1
            or t3_fac < 0.5
        ):
            return 1e6
        total = 0
        for type_, (tol, se) in subsets.items():
            maxe = 0
            n = 0
            for i in range(len(tol)):
                sp = sigma_min_from_model(
                    tol[i],
                    ns,
                    dim,
                    type_,
                    eps_mach,
                    tf=tf,
                    nsoff=nsoff,
                    u_exp=u_exp,
                    floor_c=floor_c,
                    t3_fac=t3_fac,
                )
                if sp <= MAXSIGMA + 0.5:
                    maxe = max(maxe, abs(sp - se[i]))
                    n += 1
            total += maxe
            if n < 3:
                total += 10  # penalize too few valid points
        return total

    x0 = [TOLFAC_BASE, NSOFF, U_EXP, FLOOR_C, TYPE3_FACTOR]
    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"xatol": 0.005, "fatol": 1e-5, "maxiter": 1000},
    )
    return {
        "tf": res.x[0],
        "nsoff": res.x[1],
        "u_exp": res.x[2],
        "floor_c": res.x[3],
        "t3_fac": res.x[4],
        "sum_max_err": res.fun,
        "success": res.success,
    }


# ---------------------------------------------------------------------------
# mreineck 3-term PSWF model helpers
# ---------------------------------------------------------------------------
def _pswf(c: float, x: float) -> float:
    """Normalised prolate spheroidal wave function pswf(c,x)/pswf(c,0)."""
    if abs(x) >= 1.0:
        return float("inf")
    val_x = pro_ang1(0, 0, c, x)[0]
    val_0 = pro_ang1(0, 0, c, 0.0)[0]
    if val_0 == 0.0:
        return float("inf")
    return val_x / val_0


def _mreineck_tol(
    sigma: float, ns: int, dim: int, type_: int, eps_mach: float
) -> float:
    """mreineck 3-term model: tol_kernel + tol_rounding (no phase term)."""
    tolfac = 0.18 * 1.4 ** (dim - 1) * (1.4 if type_ == 3 else 1.0)
    u = np.sqrt(1.0 - 1.0 / sigma)
    tol_kernel = tolfac * np.exp(-(ns - 1) * PI * u)
    c = PI * ns * (1.0 - 1.0 / (2.0 * sigma)) - 0.05
    pswf_arg = PI * ns / (2.0 * sigma * c)
    if pswf_arg >= 1.0:
        rdyn = float("inf")
    else:
        pv = _pswf(c, pswf_arg)
        rdyn = 1.0 / pv if pv != 0 else float("inf")
    tol_rounding = eps_mach * rdyn**dim
    return tol_kernel + tol_rounding


def _mreineck_sigma_min(
    tol: float, ns: int, dim: int, type_: int, eps_mach: float
) -> float:
    """Binary search inversion of mreineck model."""
    if _mreineck_tol(MAXSIGMA, ns, dim, type_, eps_mach) > tol:
        return MAXSIGMA + 1.0
    lo, hi = 1.01, MAXSIGMA
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _mreineck_tol(mid, ns, dim, type_, eps_mach) <= tol:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# PR comparison plot (3 models side-by-side)
# ---------------------------------------------------------------------------
def plot_pr_comparison(
    data: np.ndarray, out_png: str, model_kw: dict | None = None
) -> None:
    """Generate a 1x3 comparison figure for PR documentation.

    Columns: type 1, type 2, type 3 (double, dim=1, ns=16).
    Four curves per panel: empirical, analytical, mreineck, calibrated model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = 16
    eps_mach = 2.2e-16
    dim = 1
    prec = "d"
    kw = model_kw if model_kw else {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    for col, type_ in enumerate((1, 2, 3)):
        ax = axes[0, col]
        mask = (data["prec"] == prec) & (data["type"] == type_) & (data["dim"] == dim)
        d = data[mask]
        if len(d) == 0:
            ax.set_visible(False)
            continue

        tol = d["tol"].astype(float)
        se = d["sigma_empirical"].astype(float)
        sa = d["sigma_analytical"].astype(float)

        # mreineck model
        mr = np.array(
            [
                np.clip(_mreineck_sigma_min(t, ns, dim, type_, eps_mach), 1.0, 2.05)
                for t in tol
            ]
        )

        # calibrated model
        cal = np.array(
            [
                np.clip(
                    sigma_min_from_model(t, ns, dim, type_, eps_mach, **kw), 1.0, 2.05
                )
                for t in tol
            ]
        )

        ax.semilogx(tol, se, "b.-", ms=4, lw=1.5, label="empirical", zorder=5)
        ax.semilogx(
            tol, np.clip(sa, 1.0, 2.05), "r--", lw=1.5, label="analytical (kernel only)"
        )
        ax.semilogx(tol, mr, color="orange", ls="-.", lw=1.5, label="mreineck 3-term")
        ax.semilogx(tol, cal, "g-", lw=2, label="calibrated model")

        ax.set_xlabel(r"tolerance ($\varepsilon$)")
        ax.set_ylabel(r"$\sigma_{\min}$")
        ax.set_title(f"type {type_}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1.3, 2.15)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved to {out_png}")


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------
def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [--calibrate] <csv> [output.png]", file=sys.stderr)
        sys.exit(1)

    do_calibrate = "--calibrate" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_path = args[0]
    out_png = args[1] if len(args) > 1 else "/tmp/sigma_fit.png"

    data = load_csv(csv_path)
    combos = sorted(
        set(
            zip(
                data["prec"].tolist(),
                data["type"].tolist(),
                data["dim"].tolist(),
            )
        )
    )

    # Calibrate if requested
    model_kw = {}
    if do_calibrate:
        print("=" * 70)
        print("Calibrating model coefficients...")
        print("=" * 70)
        for prec in sorted(set(data["prec"])):
            types_avail = sorted(
                set(data["type"][(data["prec"] == prec) & (data["dim"] == 1)].tolist())
            )
            if not types_avail:
                continue
            result = calibrate(data, types=types_avail, dim=1, prec=prec)
            print(f"\n{prec} dim=1 (types {types_avail}):")
            for k, v in result.items():
                print(f"  {k:12s} = {v}")
            model_kw = {
                "tf": result["tf"],
                "nsoff": result["nsoff"],
                "u_exp": result["u_exp"],
                "floor_c": result["floor_c"],
                "t3_fac": result["t3_fac"],
            }
        print()

    # Compare model vs empirical
    print("=" * 70)
    print("Model comparison")
    print("=" * 70)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = []
    for prec, type_, dim in combos:
        mask = (data["prec"] == prec) & (data["type"] == type_) & (data["dim"] == dim)
        d = data[mask]
        if len(d) == 0:
            continue
        tol = d["tol"].astype(float)
        se = d["sigma_empirical"].astype(float)
        ns = 16 if prec == "d" else 8
        eps_mach = 2.2e-16 if prec == "d" else 1.2e-7

        pred = np.array(
            [
                min(
                    sigma_min_from_model(
                        t, ns, int(dim), int(type_), eps_mach, **model_kw
                    ),
                    2.05,
                )
                for t in tol
            ]
        )
        valid = pred <= MAXSIGMA + 0.5
        if valid.sum() > 0:
            diff = pred[valid] - se[valid]
            maxe = np.max(np.abs(diff))
            cons = np.sum(diff > 0.01)
            opt = np.sum(diff < -0.01)
        else:
            maxe = float("nan")
            cons = opt = 0

        combo = f"{prec} type={type_} dim={dim}"
        print(
            f"\n{combo}: max|err|={maxe:.4f}, cons={cons}, opt={opt}, n={valid.sum()}"
        )
        plot_data.append((combo, tol, se, pred, type_))

    # Plot
    n = len(plot_data)
    if n == 0:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, (combo, tol, se, pred, type_) in enumerate(plot_data):
        ax = axes[idx // cols, idx % cols]
        ax.semilogx(tol, se, "b.-", ms=4, lw=2, label="empirical", zorder=5)
        ax.semilogx(tol, np.clip(pred, 1.0, 2.05), "g-", lw=2, label="model")
        ax.set_xlabel("tol")
        ax.set_ylabel("sigma_min")
        ax.set_title(combo)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1.3, 2.15)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nPlot saved to {out_png}")

    # PR comparison plot (3 models side-by-side)
    cmp_png = out_png.replace(".png", "_comparison.png")
    plot_pr_comparison(data, cmp_png, model_kw=model_kw if model_kw else None)


if __name__ == "__main__":
    main()
