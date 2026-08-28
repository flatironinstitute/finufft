import math
import tempfile
from dataclasses import dataclass, fields
from numbers import Number
from pathlib import Path

import matplotlib.pyplot as plt


def pretty_number(value):
    """A size or a tolerance is a power of ten, so it prints as 1e7 or 2e-3.

    Below 1e2 the plain form is the short one, and a mantissa that is not whole
    (2.5e6) reads worse in exponent form, so both keep the plain form.
    """
    if not isinstance(value, float):
        return value
    exponent = math.floor(math.log10(abs(value))) if value else 0
    mantissa = f"{value / 10.0**exponent:g}"
    if abs(exponent) >= 2 and float(mantissa).is_integer():
        return f"{mantissa}e{exponent}"
    return int(value) if value.is_integer() else value


@dataclass
class Params:
    prec: str = "f"
    N1: Number = 320
    N2: Number = 320
    N3: Number = 1
    ntransf: int = 1
    threads: int = 1
    M: Number = 1e6
    tol: float = 1e-5

    def ndim(self) -> int:
        if self.N3 > 1:
            return 3
        if self.N2 > 1:
            return 2
        return 1

    def args(self) -> list[str]:
        return [
            f"--{f.name}={pretty_number(getattr(self, f.name))}" for f in fields(self)
        ]

    def digest_args(self) -> list[str]:
        """The case's identity, for a figure filename that must stay stable.

        The raw field values, so a change to how a case prints never renames a
        published figure and orphans the URLs already in the docs.
        """
        return [f"--{f.name}={getattr(self, f.name)}" for f in fields(self)]

    def pretty_string(self) -> str:
        n = 4
        fvalues = [
            f"{f.name}:{pretty_number(getattr(self, f.name))}" for f in fields(self)
        ]
        chunks = [" ".join(fvalues[i : i + n]) for i in range(0, len(fvalues), n)]
        return "\n".join(chunks)


# The GPU has no cores of its own to scale by, so it plays a whole-node case at
# the count the case was sized for.
GPU_WHOLE_NODE_CORES = 124


def cpu_args(param: Params, phys_cores: int) -> list[str]:
    """The case's arguments, with a whole-node case's M at per-core x phys_cores."""
    args = param.args()
    if param.threads == 0:
        i = next(i for i, f in enumerate(fields(param)) if f.name == "M")
        args[i] = f"--M={int(param.M * phys_cores)}"
    return args


# Runs inside one invocation. An arm's round is the best of NRUNS runs, and
# only a whole invocation alternates with the other binary, so inner runs buy
# no interleaving.
NRUNS = 5
# The column both harnesses report and both halves read. Interference only ever
# makes a run slower, so the fastest of NRUNS is the one least polluted by it.
METRIC_COLUMN = "min(ms)"
# Interleaved rounds per case, and the only axis that alternates the binaries,
# so a noisy case buys accuracy here. Even by requirement: an odd count leaves
# one binary a run position ahead, which a drifting machine turns into a bias.
ROUNDS = 8
assert ROUNDS % 2 == 0, "an odd ROUNDS biases the ratio"
# The sentence both halves close their summary with: one wording, because both
# halves measure the same way.
PROVENANCE = (
    "_Ratio is master/PR-head time: >1 means the PR is faster. Time is makeplan "
    "plus setpts plus execute on both halves; the GPU host transfers stage the "
    "harness's own test data, so no library change moves them and they are left "
    "out. Each case runs "
    f"{ROUNDS} rounds with the two binaries interleaved and their order "
    "alternating; each arm is tabulated at its median round and the band spans "
    "the per-round ratios. A ratio is bold where the band excludes 1.00, which "
    "is where the run resolved a change; every other row resolved nothing. "
    "Every option a caller may leave alone is left alone (sorting, upsampling "
    "factor, kernel choice, and on the GPU the spreading method), so a change "
    "to one of finufft's heuristics shows "
    "up here as the change in time it causes. The thread count is the exception: "
    "a case is defined by the count it runs at._"
)
# Single-precision tols are kept above the rounding floor eps_round ~ eps_mach*N
# so the achievability guard (check_sigma) does not reject them at the auto-
# selected upsampfac: 1e4 modes -> 2e-3, 320^2 -> 1e-4.

# A pinned case's M is absolute; a whole-node case's M is per core, which the
# runner multiplies by the node's physical core count.
PARAM_LIST = [
    Params("f", 1e4, 1, 1, 1, 1, 1e7, 2e-3),
    Params("d", 1e4, 1, 1, 1, 1, 1e7, 1e-9),
    Params("f", 320, 320, 1, 1, 1, 1e7, 1e-4),
    Params("d", 320, 320, 1, 1, 1, 1e7, 1e-9),
    Params("f", 320, 320, 1, 1, 0, 3e5, 1e-4),
    Params("d", 192, 192, 128, 1, 0, 8e4, 1e-7),
]
TRANSFORMS = [1, 2, 3]

# Every plot reads these, so a stage keeps its color across all of them. The
# stages stack bottom-up in this order.
CPU_STAGES = ["makeplan", "setpts", "execute"]
STAGE_COLORS = {
    "execute": "C0",
    "setpts": "C1",
    "makeplan": "C2",
}


def reduce_rounds(rounds: list[tuple], total) -> tuple:
    """Reduce one case's rounds to (master, pr head, band low, band high).

    ``rounds`` holds one ``(master, pr head)`` pair per round, and ``total``
    returns one arm's time. Each arm is tabulated at its median round and the
    ratio is the medians' ratio: on a null change that point estimate errs by
    ~0.2-0.4% where the fastest-round pair errd by up to 4% (measured across
    builds 32 and 47), and interference runs both arms down the round the same
    way. Least-polluted band unchanged: it still spans the per-round ratios.
    """
    ratios = [total(master) / total(prhead) for master, prhead in rounds]
    middle = len(rounds) // 2 - 1
    by_master = sorted(rounds, key=lambda r: total(r[0]))
    by_prhead = sorted(rounds, key=lambda r: total(r[1]))
    return (by_master[middle][0], by_prhead[middle][1], min(ratios), max(ratios))


def case_table(rows: list[tuple]) -> list[str]:
    """The per-case block both halves publish: same columns, same order.

    Each row is ``(label, master ms, pr head ms, band low, band high)``. A band
    that brackets 1.00 resolved no change, however far from 1.00 the ratio fell,
    so only a band that excludes 1.00 puts its ratio in bold.
    """
    lines = [
        "| case | master (ms) | PR head (ms) | ratio | band |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, master, prhead, low, high in rows:
        ratio = f"{master / prhead:.2f}x"
        # Against the printed band, so a reader can check the bold by eye.
        if not round(low, 2) <= 1.0 <= round(high, 2):
            ratio = f"**{ratio}**"
        lines.append(
            f"| {name.replace(chr(10), ' ')} | {master:.2f} | {prhead:.2f} | "
            f"{ratio} | {low:.2f}-{high:.2f} |"
        )
    return lines


def stacked_grid(cells: list[tuple], suptitle: str, output) -> None:
    """The plot both halves publish: one stacked pair per case, transforms across.

    Each cell is ``(title, master stage times, pr head stage times)``, keyed by
    stage in ms, laid out row major with one column per transform. The bar label
    is the stack's own ratio, so it matches both the bar it sits on and the
    table; the band is in the table.
    """
    ncols = len(TRANSFORMS)
    nrows = -(-len(cells) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
    for ax, (title, master, prhead) in zip(axs.flat, cells):
        bottoms = [0.0, 0.0]
        bars = None
        for stage in CPU_STAGES:
            heights = [master[stage], prhead[stage]]
            bars = ax.bar(
                ["master", " pr head"],
                heights,
                bottom=bottoms,
                label=stage,
                color=STAGE_COLORS[stage],
            )
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax.set_ylim(top=max(bottoms) * 1.15)
        ax.bar_label(bars, labels=["1.00x", f"{bottoms[0] / bottoms[1]:.2f}x"])
        ax.set_ylabel("time (ms)")
        ax.set_title(title)
    # One legend for the grid, between title and axes: inside an axes it covers
    # the bars. The offsets are inches from the top, so one set of numbers holds
    # at any row count.
    height = fig.get_figheight()
    fig.legend(
        *axs.flat[0].get_legend_handles_labels(),
        ncol=len(CPU_STAGES),
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.85 / height),
    )
    fig.suptitle(suptitle, fontsize=24, y=1 - 0.35 / height, va="center")
    fig.tight_layout(pad=2, h_pad=2, rect=(0, 0, 1, 1 - 1.15 / height))
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # The bands of the widest and the tightest case of a null run, which must
    # light nothing up, and one resolved gain and one resolved regression.
    null = [("wide", 53.62, 54.04, 0.65, 1.52), ("tight", 1151.48, 1148.81, 1.0, 1.0)]
    real = [("gain", 120.0, 100.0, 1.15, 1.25), ("loss", 100.0, 125.0, 0.78, 0.83)]
    table = case_table(null + real)
    assert all("**" not in row for row in table[2:4]), table[2:4]
    assert "**1.20x**" in table[4], table[4]
    assert "**0.80x**" in table[5], table[5]
    # A band whose ends round onto 1.00 prints as 1.00 and must not go bold.
    assert "**" not in case_table([("edge", 1.0, 1.0, 1.001, 1.02)])[2]
    # A whole-node case's M is per core and multiplies by the node; a pinned
    # case's never moves.
    whole = Params("d", 192, 192, 128, 1, 0, 8e4, 1e-7)
    pinned = Params("f", 1e4, 1, 1, 1, 1, 1e7, 2e-3)
    assert "--M=80000" in cpu_args(whole, 1), cpu_args(whole, 1)
    assert "--M=9920000" in cpu_args(whole, 124), cpu_args(whole, 124)
    assert "--M=1e7" in cpu_args(pinned, 31), cpu_args(pinned, 31)
    # One stacked cell per transform, so the grid is one row of len(TRANSFORMS).
    stage_ms = {stage: 1.0 for stage in CPU_STAGES}
    cells = [(f"type {t}", stage_ms, stage_ms) for t in TRANSFORMS]
    out = Path(tempfile.gettempdir()) / "perftest_config_selfcheck.svg"
    stacked_grid(cells, "self check", out)
    assert out.stat().st_size > 0, out
    out.unlink()
    print("\n".join(table))
    print("case_table ok: bold only where the printed band excludes 1.00")
