"""The CPU half of the perftest comment: master against the PR-head build.

``cpu_compare`` measures every case, writes the plot and returns the facts the
comment quotes. ``tools/perf-comment.py`` is the caller.
"""

import sys
import time
from pathlib import Path

from perftest_config import (
    CPU_STAGES,
    PARAM_LIST,
    ROUNDS,
    TRANSFORMS,
    reduce_rounds,
    stacked_grid,
)
from perftest_helpers import (
    METRIC_COLUMN,
    build_command,
    cpu_metadata,
    measure_cases,
    physical_cores,
    read_cmake_metadata,
    run_perftest,
    usable_ncores,
)


def total(df) -> float:
    """One arm's time for a case: the stages the plot stacks, summed."""
    return float(df.loc[CPU_STAGES, METRIC_COLUMN].sum())


def binaries(items: list[str]) -> dict[str, str]:
    """``FFT=PATH`` arguments into a dict keyed by FFT backend.

    Keyed rather than ordered: the two arms have to pair backend for backend,
    and a name says so where a position only implies it.
    """
    return dict(item.split("=", 1) for item in items)


def cpu_compare(master_items: list[str], pr_items: list[str], plot_output) -> dict:
    """Measure every CPU case, write the plot, and return the comment's facts.

    One binary of each arm per FFT backend. Both backends read the same finufft
    heuristics -- one upsampfac model serves them -- so a retune can move them
    in opposite directions, which one backend alone cannot show.
    """
    master_bins = binaries(master_items)
    prhead_bins = binaries(pr_items)
    assert master_bins.keys() == prhead_bins.keys(), "the arms carry other backends"

    facts = cpu_metadata()
    facts["backends"] = " ".join(master_bins)
    facts["ncores"] = usable_ncores()
    # The count a --threads=0 case runs with: finufft asks for the physical
    # cores of the affinity mask, which on an SMT part is half the processors.
    facts["ncores_phys"] = len(physical_cores())
    # Every backend is built by the same compiler with the same flags, so one
    # tree answers for all of them.
    facts.update(
        read_cmake_metadata(Path(next(iter(master_bins.values()))).parent.parent)
    )

    # A case is a parameter set, a backend and a transform. Backend is the inner
    # row so a parameter set's two backends sit next to each other in the plot.
    grid = [
        (param, fft, transform)
        for param in PARAM_LIST
        for fft in master_bins
        for transform in TRANSFORMS
    ]

    def measure(k: int, cpu: int | None):
        param, fft, transform = grid[k]
        t0 = time.monotonic()
        master_cmd = build_command(param, transform, master_bins[fft], cpu)
        prhead_cmd = build_command(param, transform, prhead_bins[fft], cpu)
        # The two binaries are interleaved and their order alternates:
        # whichever runs first pays any residual first-run cost, so a fixed
        # order biases the ratio.
        rounds = []
        spread = param.threads != 1
        for r in range(ROUNDS):
            if r % 2:
                master_df = run_perftest(master_cmd, spread)
                prhead_df = run_perftest(prhead_cmd, spread)
            else:
                prhead_df = run_perftest(prhead_cmd, spread)
                master_df = run_perftest(master_cmd, spread)
            rounds.append((master_df, prhead_df))
        master_df, prhead_df, low, high = reduce_rounds(rounds, total)
        # One line per case, printed when the case ends: concurrent cases would
        # interleave a start line and a finish line beyond reading. The
        # per-round ratios are what a wide band is made of.
        ratios = " ".join(f"{total(m) / total(p):.2f}" for m, p in rounds)
        print(
            f"[{k + 1}/{len(grid)}] fft={fft} type={transform} "
            f"{param.pretty_string().replace(chr(10), ' ')} on cpu {cpu} "
            f"done in {time.monotonic() - t0:.1f}s, "
            f"{total(master_df) / total(prhead_df):.2f}x band {low:.2f}-{high:.2f} "
            f"rounds {ratios}",
            file=sys.stderr,
            flush=True,
        )
        return " ".join(master_cmd), master_df, prhead_df, low, high

    cells, rows, cmd_list = [], [], []
    results = measure_cases([param for param, _, _ in grid], measure)
    for (param, fft, transform), result in zip(grid, results):
        master_cmd, master_df, prhead_df, low, high = result
        cmd_list.append(master_cmd)
        label = f"fft:{fft} type:{transform} {param.pretty_string()}"
        rows.append((label, total(master_df), total(prhead_df), low, high))
        cells.append(
            (
                label,
                {stage: master_df.loc[stage, METRIC_COLUMN] for stage in CPU_STAGES},
                {stage: prhead_df.loc[stage, METRIC_COLUMN] for stage in CPU_STAGES},
            )
        )
    stacked_grid(
        cells, "Performance change between master and latest pr HEAD", plot_output
    )

    facts["commands"] = "\n".join(cmd_list)
    facts["rows"] = rows
    return facts
