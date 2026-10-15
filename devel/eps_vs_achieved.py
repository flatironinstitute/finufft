#!/usr/bin/env python3
"""Plot eps (tol) vs achieved relative L2 error from devel/eps_vs_achieved CSV.

Usage: eps_vs_achieved.py CSV PREC OUT.png
PREC is float or double; filters rows, draws a 3x3 grid (row = transform type
1/2/3, column = dimension 1/2/3). Each cell holds two data curves, sigma=auto
and sigma=2.0, log-log, with the identity line y=tol, the slack line
y = tolslack[type-1]*tol used by test/tolsweep.cpp, and the per-series
rounding floor. x and y axes are shared so cells are comparable.
"""
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

USAGE = f"usage: {sys.argv[0]} CSV PREC OUT.png  (PREC: float|double)"
if len(sys.argv) != 4 or sys.argv[2] not in ("float", "double"):
    sys.exit(USAGE)
csv, prec, out = sys.argv[1:4]

# test/tolsweep.cpp line 36: tolslack per type.
TOLSLACK = {1: 4.0, 2: 4.0, 3: 5.0}

# test/tolsweep.cpp lines 50-56: floor[nu][3], row = upsampfac index
# (0 = auto, 1 = 1.25, 2 = 2.0), column = dim - 1. These are tolsweep's
# acceptance floors, not the library's rounding_floor() model: the auto
# series can pick a different grid at each tol.
FLOOR = {
    "float": {0: (2e-5, 2e-5, 1e-5), 2: (2e-5, 2e-5, 1e-5)},
    "double": {0: (3e-14, 3e-14, 3e-14), 2: (3e-14, 3e-14, 3e-14)},
}

rows = []
with open(csv) as f:
    next(f)  # header
    for line in f:
        p, d, t, s, tol, err, ier = line.strip().split(",")
        if p != prec:
            continue
        rows.append((int(d), int(t), float(s), float(tol), float(err), int(ier)))

if not rows:
    sys.exit(f"no rows for prec={prec} in {csv}")

sigmas = sorted({r[2] for r in rows})
types = (1, 2, 3)
dims = (1, 2, 3)
# per-sigma colors; cell already encodes (type, dim), color encodes sigma
sig_colors = {0.0: "C0", 2.0: "C1"}
sig_styles = {0.0: "-", 2.0: "--"}


def series(typ, dim, sig):
    sel = sorted(
        (r[3], r[4], r[5]) for r in rows
        if r[0] == dim and r[1] == typ and r[2] == sig
    )
    if not sel:
        return None
    tols = [t for t, _, _ in sel]
    errs = [e for _, e, _ in sel]
    iers = [i for _, _, i in sel]
    return tols, errs, iers


# each series needs two or more ier=0 rows; name the failing rows, before any write
bad = []
for typ in types:
    for dim in dims:
        for sig in (0.0, 2.0):
            cell = [r for r in rows if r[0] == dim and r[1] == typ and r[2] == sig]
            ok = [r for r in cell if r[5] == 0 and math.isfinite(r[4])]
            if len(ok) < 2:
                failed = [(r[3], r[5]) for r in cell if r[5] != 0]
                bad.append(f"type {typ} dim {dim} sigma={sig}: {len(ok)} ok row(s), "
                           f"failing (tol, ier): {failed}")
if bad:
    sys.exit(f"{csv}, prec={prec}:\n" + "\n".join(bad))


def draw(out_png):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    floors = FLOOR[prec]
    # shared x range from every requested tol (valid regardless of ier), before
    # drawing, so no series can lock the limits and clip another; set and
    # invert once after plotting
    xmin = min(r[3] for r in rows) * 0.5
    xmax = max(r[3] for r in rows) * 2.0
    for row, typ in zip(axes, types):
        slack = TOLSLACK[typ]
        for ax, dim in zip(row, dims):
            for sig in sigmas:
                s = series(typ, dim, sig)
                if s is None:
                    continue
                tols, errs, iers = s
                # NaN at a failed row breaks the line there instead of
                # bridging across it as if the neighbours were adjacent.
                plot_errs = [e if i == 0 and math.isfinite(e) else float("nan")
                             for e, i in zip(errs, iers)]
                ax.loglog(tols, plot_errs, sig_styles[sig], color=sig_colors[sig],
                          marker="o", markersize=3, linewidth=1)
                # first miss: largest tol at which err > tolslack*tol, found by
                # scanning tol downward as tolsweep does
                for tol, err, ier in zip(reversed(tols), reversed(errs), reversed(iers)):
                    if ier == 0 and err > slack * tol:
                        ax.plot(tol, err, "x", color=sig_colors[sig],
                                markersize=8, markeredgewidth=2)
                        break
                for tol, err, ier in zip(tols, errs, iers):
                    if ier == 0 and math.isfinite(err):
                        continue
                    print(f"failed: type {typ} dim {dim} sigma={sig} "
                          f"tol={tol:g} ier={ier}", file=sys.stderr)
                    # fixed axes-relative height: a failed row's err is NaN,
                    # so plot at a y position every tol can reach.
                    ax.plot(tol, 0.05, "s", transform=ax.get_xaxis_transform(),
                            color=sig_colors[sig], markersize=6,
                            markerfacecolor="none")
                # tolsweep acceptance floor, same color as the data curve
                floor = floors[sig][dim - 1]
                ax.loglog([xmax, xmin], [floor, floor], ":",
                          color=sig_colors[sig], linewidth=1, alpha=0.7)
            ax.loglog([xmax, xmin], [xmax, xmin], ":", color="0.4")
            ax.loglog([xmax, xmin], [slack * xmax, slack * xmin], "--",
                      color="0.4", alpha=0.6, linewidth=0.8)
            ax.grid(True, which="both", alpha=0.3)
    # set shared limits once, inverted (descending tol); sharex covers all axes
    axes[0, 0].set_xlim(xmax, xmin)
    for ax, dim in zip(axes[0], dims):
        ax.set_title(f"dim {dim}", fontsize=11)
    for ax, typ in zip(axes[:, 0], types):
        ax.set_ylabel(f"type {typ}\nachieved relative L2 error", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("requested tol (eps)")
    proxies = [
        Line2D([], [], linestyle="-", color="C0", marker="o", markersize=3,
               linewidth=1, label="sigma=auto"),
        Line2D([], [], linestyle="--", color="C1", marker="o", markersize=3,
               linewidth=1, label="sigma=2.0"),
        Line2D([], [], linestyle=":", color="C0", linewidth=1, alpha=0.7,
               label="tolsweep acceptance floor (sigma=auto)"),
        Line2D([], [], linestyle=":", color="C1", linewidth=1, alpha=0.7,
               label="tolsweep acceptance floor (sigma=2)"),
        Line2D([], [], linestyle=":", color="0.4", label="err = tol"),
        Line2D([], [], linestyle="--", color="0.4", alpha=0.6, linewidth=0.8,
               label="slack: type1/2 x4, type3 x5"),
        Line2D([], [], linestyle="none", marker="x", color="0.3",
               markersize=8, markeredgewidth=2, label="first tol miss"),
    ]
    fig.legend(handles=proxies, loc="lower center", ncol=4, fontsize=9,
               frameon=False)
    fig.suptitle(f"FINUFFT {prec}: eps vs achieved relative L2 error")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


draw(out)
