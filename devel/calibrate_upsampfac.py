#!/usr/bin/env python3
"""Fit the heuristics.hpp upsampfac cost-model constants from a calibration sweep.

Reads the per-sigma *detail* CSV emitted by devel/calibrate_upsampfac (columns:
prec,dim,type,tol,N,M,threads,sigma,execute_ms_min,is_optimal,ns,nc,grid_traffic,fft_G).
For a candidate constant set we reconstruct the model cost on each config's sigma grid
exactly (the detail CSV carries the ISA-specific grid_traffic and the fine-grid product
fft_G so no C++ formula has to be re-derived here), take its argmin sigma_model, and
minimize  Sum (sigma_model - sigma_opt)^2  over the type-1/2 configs. Type-3 sigma is
selected by a recursive model whose inner cost itself depends on the constants being fit,
so it cannot be reconstructed from a static CSV; type-3 rows are reported only.

Production model (include/finufft/heuristics.hpp) has TWO constants: C_FFT_BASE and
K_FFT_THREAD. The spread term is pure grid traffic. C_HORNER below is an EXPERIMENTAL
candidate weight on the per-point Horner work (c_horner * nc); the production model uses
C_HORNER = 0 (the term is not present). The fitter explores whether a nonzero value would
better match the empirical optima -- on the first sweep it was not warranted (and the
data was too noisy to trust; see the caveat in calibrate_upsampfac.cpp). The C_HORNER=0
baseline is printed alongside the fit so you can see what (if anything) the term buys.

Two-stage fit: fix K_FFT_THREAD=0.5 and fit {C_FFT_BASE, C_HORNER}, then refine all three.

NOTE: do not bake constants from a noisy sweep. Re-run the C++ tool with --n_runs>=15,
--threads=1, and cross-check a few cells against perftest at forced --upsampfact first.

Run: uv run --with scipy --with numpy --with matplotlib \
        devel/calibrate_upsampfac.py --detail detail.csv [--plot scatter.png]
"""

import argparse
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

COLS = [
    "prec",
    "dim",
    "type",
    "tol",
    "N",
    "M",
    "threads",
    "sigma",
    "ms",
    "is_opt",
    "ns",
    "nc",
    "grid_traffic",
    "fft_G",
]


def load_detail(path):
    """Group per-sigma rows into configs keyed by (prec,dim,type,tol,N,M,threads)."""
    configs = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("prec") or line.startswith("#"):
                continue
            p = line.split(",")
            row = dict(zip(COLS, p))
            key = (
                row["prec"],
                int(row["dim"]),
                int(row["type"]),
                float(row["tol"]),
                int(row["N"]),
                int(row["M"]),
                int(row["threads"]),
            )
            configs[key].append(
                dict(
                    sigma=float(row["sigma"]),
                    ms=float(row["ms"]),
                    is_opt=int(row["is_opt"]),
                    ns=int(row["ns"]),
                    nc=int(row["nc"]),
                    grid_traffic=float(row["grid_traffic"]),
                    fft_G=float(row["fft_G"]),
                )
            )
    for k in configs:
        configs[k].sort(key=lambda r: r["sigma"])
    return configs


def model_cost(rows, dim, M, threads, c_fft_base, k_thread, c_horner):
    """Model cost per sigma row, mirroring include/finufft/heuristics.hpp."""
    c_fft = c_fft_base * max(1, threads) ** k_thread
    out = []
    for r in rows:
        spread = M * (r["grid_traffic"] + c_horner * r["nc"]) * r["ns"] ** (dim - 1)
        G = r["fft_G"]
        fft = c_fft * G * math.log2(G) if G > 1 else 0.0
        out.append(spread + fft)
    return out


def sigma_opt(rows):
    for r in rows:
        if r["is_opt"]:
            return r["sigma"]
    return min(rows, key=lambda r: r["ms"])["sigma"]


def sigma_model(rows, dim, M, threads, params):
    costs = model_cost(rows, dim, M, threads, *params)
    kmin = int(np.argmin(costs))  # smallest-sigma tie-break via stable argmin
    return rows[kmin]["sigma"]


def objective(params, t12_configs, fixed_k):
    c_fft_base, c_horner = params if fixed_k is not None else (params[0], params[2])
    k_thread = fixed_k if fixed_k is not None else params[1]
    p = (c_fft_base, k_thread, c_horner)
    err = 0.0
    for (prec, dim, typ, tol, N, M, threads), rows in t12_configs:
        sm = sigma_model(rows, dim, M, threads, p)
        err += (sm - sigma_opt(rows)) ** 2
    return err


def report(configs, params):
    """Per-(dim,type) argmin residual under fitted params + measured slowdown."""
    buckets = defaultdict(list)
    for key, rows in configs.items():
        prec, dim, typ, tol, N, M, threads = key
        so = sigma_opt(rows)
        sm = sigma_model(rows, dim, M, threads, params)
        # measured slowdown of the model's sigma vs the empirical optimum
        ms_by_sigma = {round(r["sigma"], 4): r["ms"] for r in rows}
        t_opt = min(r["ms"] for r in rows)
        t_model = ms_by_sigma.get(round(sm, 4), t_opt)
        buckets[(dim, typ)].append(
            (abs(sm - so), t_model / t_opt if t_opt > 0 else 1.0)
        )
    print(
        f"\n{'dim':>3} {'type':>4} {'n':>3} {'mean|dsig|':>10} {'max|dsig|':>9} "
        f"{'mean slow':>10} {'max slow':>9}"
    )
    for key in sorted(buckets):
        b = buckets[key]
        ds = [x[0] for x in b]
        sl = [x[1] for x in b]
        print(
            f"{key[0]:>3} {key[1]:>4} {len(b):>3} {np.mean(ds):>10.3f} "
            f"{np.max(ds):>9.3f} {np.mean(sl):>10.3f} {np.max(sl):>9.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", required=True, help="per-sigma detail CSV")
    ap.add_argument("--plot", default="", help="optional scatter PNG path")
    args = ap.parse_args()

    configs = load_detail(args.detail)
    if not configs:
        raise SystemExit("no rows in detail CSV")
    t12 = [(k, v) for k, v in configs.items() if k[2] in (1, 2)]
    print(f"loaded {len(configs)} configs ({len(t12)} type-1/2 used for the fit)")
    if not t12:
        raise SystemExit("no type-1/2 configs to fit")

    # production baseline: current heuristics.hpp constants, no Horner term.
    print("\nproduction baseline (C_FFT_BASE=9.0, K_FFT_THREAD=0.5, C_HORNER=0):")
    report(configs, (9.0, 0.5, 0.0))

    # stage 1: fix K_FFT_THREAD=0.5, fit {C_FFT_BASE, C_HORNER} (Nelder-Mead: argmin
    # objective is piecewise-constant in params, so no gradient method). C_HORNER lower
    # bound is 0 so the fit can decline the experimental term.
    r1 = minimize(
        objective,
        x0=[9.0, 0.0],
        args=(t12, 0.5),
        method="Nelder-Mead",
        options=dict(xatol=1e-3, fatol=1e-6, maxiter=4000),
    )
    c_fft_base, c_horner = np.clip(r1.x, [1, 0.0], [30, 5])
    # stage 2: refine all three
    r2 = minimize(
        objective,
        x0=[c_fft_base, 0.5, c_horner],
        args=(t12, None),
        method="Nelder-Mead",
        options=dict(xatol=1e-3, fatol=1e-6, maxiter=8000),
    )
    c_fft_base, k_thread, c_horner = np.clip(r2.x, [1, 0.1, 0.0], [30, 0.9, 5])
    params = (c_fft_base, k_thread, c_horner)

    print("\n=== fitted constants (C_HORNER is experimental; production uses 0) ===")
    print(f"constexpr double C_FFT_BASE   = {c_fft_base:.2f};")
    print(f"constexpr double K_FFT_THREAD = {k_thread:.2f};")
    print(f"constexpr double C_HORNER     = {c_horner:.2f};")

    print(
        "\nper-(dim,type) residuals under fitted constants "
        "(type 3 = report only, validate by re-running C++):"
    )
    report(configs, params)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        so, sm = [], []
        for key, rows in configs.items():
            _, dim, _, _, _, M, threads = key
            so.append(sigma_opt(rows))
            sm.append(sigma_model(rows, dim, M, threads, params))
        plt.figure(figsize=(5, 5))
        plt.scatter(so, sm, s=12, alpha=0.6)
        plt.plot([1.1, 2.05], [1.1, 2.05], "k--", lw=0.8)
        plt.xlabel("sigma_opt (empirical)")
        plt.ylabel("sigma_model (fitted)")
        plt.title("upsampfac: model vs optimum")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120)
        print(f"\nwrote scatter to {args.plot}")


if __name__ == "__main__":
    main()
