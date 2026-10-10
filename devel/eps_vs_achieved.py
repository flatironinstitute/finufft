#!/usr/bin/env python3
"""Plot eps (tol) vs achieved relative L2 error from devel/eps_vs_achieved CSV.

Usage: eps_vs_achieved.py CSV PREC OUT.png
PREC is float or double; filters rows, draws a 3x3 grid (row = transform type
1/2/3, column = dimension 1/2/3). Each cell holds two data curves, sigma=auto
and sigma=2.0, log-log, with the identity line y=tol, the slack line
y = tolslack[type-1]*tol used by test/tolsweep.cpp, and the per-series
rounding floor. x and y axes are shared so cells are comparable.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

csv, prec, out = sys.argv[1], sys.argv[2], sys.argv[3]

# test/tolsweep.cpp line 36: tolslack per type.
TOLSLACK = {1: 4.0, 2: 4.0, 3: 5.0}

# test/tolsweep.cpp lines 50-56: floor[nu][3], row = upsampfac index
# (0 = auto, 1 = 1.25, 2 = 2.0), column = dim - 1.
FLOOR = {
    "float": {0: (1e-4, 1e-4, 2e-4), 2: (2e-5, 2e-5, 1e-5)},
    "double": {0: (1e-9, 2e-9, 3e-8), 2: (3e-14, 3e-14, 3e-14)},
}
SIG_IDX = {0.0: 0, 2.0: 2}  # series sigma -> upsampfac slot in FLOOR

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
        (r[3], r[4]) for r in rows
        if r[0] == dim and r[1] == typ and r[2] == sig and r[5] == 0
    )
    if not sel:
        return None
    tols = [t for t, _ in sel]
    errs = [e for _, e in sel]
    return tols, errs


# each series needs two or more ier=0 rows; name the failing rows, before any write
bad = []
for typ in types:
    for dim in dims:
        for sig in (0.0, 2.0):
            cell = [r for r in rows if r[0] == dim and r[1] == typ and r[2] == sig]
            ok = [r for r in cell if r[5] == 0]
            if len(ok) < 2:
                failed = [(r[3], r[5]) for r in cell if r[5] != 0]
                bad.append(f"type {typ} dim {dim} sigma={sig}: {len(ok)} ok row(s), "
                           f"failing (tol, ier): {failed}")
if bad:
    sys.exit(f"{csv}, prec={prec}:\n" + "\n".join(bad))


def draw(out_png):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    floors = FLOOR[prec]
    for row, typ in zip(axes, types):
        slack = TOLSLACK[typ]
        siglab = {0.0: "sigma=auto", 2.0: "sigma=2.0"}
        automs = "first tol miss (auto)"
        ms2m = "first tol miss (2.0)"
        misslab = {0.0: automs, 2.0: ms2m}
        for ax, dim in zip(row, dims):
            for sig in sigmas:
                s = series(typ, dim, sig)
                if s is None:
                    continue
                tols, errs = s
                ax.loglog(tols, errs, sig_styles[sig], color=sig_colors[sig],
                          label=siglab[sig], marker="o", markersize=3,
                          linewidth=1)
                # first miss: largest tol at which err > tolslack*tol, found by
                # scanning tol downward as tolsweep does
                for tol, err in zip(reversed(tols), reversed(errs)):
                    if err > slack * tol:
                        ax.plot(tol, err, "x", color=sig_colors[sig],
                                markersize=8, markeredgewidth=2,
                                label=misslab[sig])
                        break
                # per-series floor, same color as the data curve
                floor = floors[SIG_IDX[sig]][dim - 1]
                lims = ax.get_xlim()
                ax.loglog(lims, [floor, floor], ":", color=sig_colors[sig],
                          linewidth=1, alpha=0.7,
                          label=f"floor ({siglab[sig]}, dim {dim})")
                ax.set_xlim(lims)
            lims = ax.get_xlim()
            ax.loglog(lims, lims, ":", color="0.4", label="err = tol")
            ax.loglog(lims, [slack * x for x in lims], "--", color="0.4",
                      alpha=0.6, linewidth=0.8, label=f"slack x{slack:g} (type {typ})")
            ax.set_xlim(lims)
            ax.invert_xaxis()
            ax.grid(True, which="both", alpha=0.3)
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
               label="floor (sigma=auto, per dim)"),
        Line2D([], [], linestyle=":", color="C1", linewidth=1, alpha=0.7,
               label="floor (sigma=2, per dim)"),
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
    # assert grid shape and that every artist is named, before the write
    assert len(axes.flat) == 9
    for ax in axes.flat:
        for lab in (l.get_label() for l in ax.get_lines()):
            assert lab and not lab.startswith("_"), f"unlabeled artist: {lab!r}"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"assertions ok: 9 axes, all artists labeled")
    plt.close(fig)
    print(f"wrote {out_png}")


draw(out)
