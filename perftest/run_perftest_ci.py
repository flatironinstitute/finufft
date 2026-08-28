#!/usr/bin/env python3
"""Run the perftest matrix across tagged builds and render one page section."""

# /// script
# dependencies = ["matplotlib", "pandas", "numpy", "jinja2", "py-cpuinfo", "archspec"]
# ///

import argparse
import hashlib
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from cuperftest_helpers import (
    GPU_METHOD_NAMES,
    gpu_args,
    gpu_cases,
    gpu_method_heading,
    gpu_methods,
    gpu_params_string,
    nvcc_version,
    query_gpu,
    run_cuperftest,
)
from perftest_config import CPU_STAGES, PARAM_LIST, STAGE_COLORS, TRANSFORMS
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

# The GPU library is one backend rather than a choice of FFT libraries, so the
# backend name is the language it is written in.
CUDA = "cuda"


def cpu_times(
    binary: Path, param, transform: int, method: int | None, cpu: int | None
) -> dict[str, float]:
    return {
        stage: run_perftest(
            build_command(param, transform, str(binary), cpu), param.threads != 1
        ).loc[stage, METRIC_COLUMN]
        for stage in CPU_STAGES
    }


def gpu_times(
    binary: Path, param, transform: int, method: int | None, cpu: int | None
) -> dict[str, float]:
    return run_cuperftest(str(binary), gpu_args(param, transform, method))


def measure(
    times,
    binary: Path,
    param,
    transform: int,
    method: int | None,
    tag: str,
    cpu: int | None,
):
    """One tag's stage times, or None with a reason on stderr.

    A release predates what a case asks of it - cuperftest grew type 3 after
    every tag on the plot - so a tag that cannot run a case drops out of that
    case's plot instead of failing the page. Silence would read as coverage, so
    the reason is printed.
    """
    if not binary.exists():
        print(f"  tag={tag} skipped: no {binary}", file=sys.stderr, flush=True)
        return None
    try:
        return times(binary, param, transform, method, cpu)
    except (RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"  tag={tag} skipped: {exc}", file=sys.stderr, flush=True)
        return None


def cpu_facts(builds_root: Path, cache_tag: str) -> list[tuple[str, str]]:
    cpu = cpu_metadata()
    meta = read_cmake_metadata(builds_root / cache_tag)
    return [
        ("CPU", cpu["cpu_name"]),
        ("Arch", cpu["arch"]),
        ("Usable processors", usable_ncores()),
        # What a threads=0 case runs with: finufft asks for the physical cores
        # of the affinity mask, which on an SMT part is half the processors.
        ("Usable physical cores", len(physical_cores())),
        ("Microarchitecture", cpu["uarch"]),
        ("psABI level", cpu["level"]),
        ("Compiler", meta["compiler_version"]),
        ("Compiler flags", meta["compiler_flags"]),
    ]


def gpu_facts(builds_root: Path, cache_tag: str) -> list[tuple[str, str]]:
    meta = read_cmake_metadata(builds_root / cache_tag)
    release = next(
        (ln.strip() for ln in nvcc_version().splitlines() if "release" in ln), "NA"
    )
    return [
        ("Device", query_gpu()),
        ("Toolkit", release),
        ("Host compiler", meta["compiler_version"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run perftest matrix and generate plots."
    )
    parser.add_argument(
        "--backend", default="fftw", help=f"fftw, ducc, or {CUDA} for the GPU library."
    )
    parser.add_argument("--builds-root", default="./builds")
    parser.add_argument(
        "--tag-list",
        default="",
        help="Space-separated tags in preferred display order.",
    )
    parser.add_argument(
        "--page-template",
        default="docs/performance_section.rst.j2",
        help="Path to the docs template page to render.",
    )
    parser.add_argument(
        "--output",
        default="./outputs",
        help="Output directory for generated performance report.",
    )
    parser.add_argument(
        "--cmake-cache-from",
        default="master",
        help="Tag whose CMakeCache.txt is used for compiler metadata.",
    )
    args = parser.parse_args()

    builds_root = Path(args.builds_root)
    assert builds_root.exists(), f"builds_root {builds_root} does not exist"
    output_dir = Path(args.output)
    assert output_dir.exists(), f"output dir {output_dir} does not exist"
    template_path = Path(args.page_template)
    assert template_path.exists(), f"template {template_path} does not exist"

    tags = args.tag_list.split()

    # One switch, so the two libraries share the loop, the figures and the page
    # section they render into. The GPU library has no thread count, so its case
    # list is the same one with that distinction collapsed. Both libraries plot
    # CPU_STAGES: cuperftest also times the host-device transfers, but those
    # stage the harness's own test data, so no library change moves them and a
    # release-over-release plot of them reads the node's link.
    gpu = args.backend == CUDA
    cases = gpu_cases() if gpu else PARAM_LIST
    binary = Path("perftest/cuda/cuperftest") if gpu else Path("perftest/perftest")
    times = gpu_times if gpu else cpu_times
    case_string = (
        gpu_params_string if gpu else (lambda param, method: param.pretty_string())
    )
    heading = "cuFFT backend" if gpu else f"{args.backend.upper()} backend"

    dim_transform_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # The GPU library spreads by one of several methods and the library's own
    # pick is one of them, so a case is charted once per method it can run: a
    # chart that mixed them would hide which one a release moved. The CPU
    # library has no such axis, so its grid keeps one chart per case.
    grid = [
        (param, transform, method)
        for param in cases
        for transform in TRANSFORMS
        for method in (gpu_methods(transform) if gpu else [None])
    ]

    def measure_case(k: int, cpu: int | None):
        param, transform, method = grid[k]
        t0 = time.monotonic()
        x: list[str] = []
        series: dict[str, list[float]] = {stage: [] for stage in CPU_STAGES}
        for tag in tags:
            tag_times = measure(
                times, builds_root / tag / binary, param, transform, method, tag, cpu
            )
            if tag_times is None:
                continue
            x.append(tag)
            for stage in CPU_STAGES:
                series[stage].append(tag_times[stage])
        # One line per case, printed when the case ends: concurrent cases would
        # interleave a start line, a line per tag and a finish line beyond
        # reading. It names the tags that ran, since a skip is silent here.
        print(
            f"[{k + 1}/{len(grid)}] type={transform} "
            f"{case_string(param, method).replace(chr(10), ' ')} on cpu {cpu} "
            f"done in {time.monotonic() - t0:.1f}s, tags={' '.join(x)}",
            file=sys.stderr,
            flush=True,
        )
        return x, series

    # The GPU cases keep a thread count they do not use, and one device cannot
    # hold two of them at once, so that half stays one case at a time.
    results = measure_cases([param for param, _, _ in grid], measure_case, not gpu)
    for (param, transform, method), (x, series) in zip(grid, results):
        if len(x) < 2:
            continue

        fig, ax = plt.subplots()
        # The shared colors, not the default cycle: a stage keeps its color
        # across this page and both plots in the PR comment. Bottom-up in
        # the stage order, which is the order the two plots stack in.
        ax.stackplot(
            x,
            *(series[stage] for stage in CPU_STAGES),
            labels=CPU_STAGES,
            colors=[STAGE_COLORS[stage] for stage in CPU_STAGES],
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Version")
        ax.set_ylabel("Min time (ms)")
        ax.legend()
        # The figure names the method itself, so a chart read on its own - the
        # page links each one by URL - still says which method it timed.
        if method is not None:
            ax.set_title(
                f"type {transform}, method {method} ({GPU_METHOD_NAMES[method]})"
            )

        # A digest of the case, so the filename is stable across runs and
        # the raw.githubusercontent URLs already published keep resolving
        # once the perftest-results branch is force-pushed.
        # A method suffix only on the backend that has methods, so the CPU page's
        # figures keep the names their published URLs already point at.
        method_key = "" if method is None else f"|m{method}"
        key = f"{args.backend}|t{transform}{method_key}|" + "|".join(
            param.digest_args()
        )
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        file = f"perftestci_{digest}.svg"
        durations = np.sum([series[stage] for stage in CPU_STAGES], axis=0)
        ax.set_ylim(top=np.max(durations) * 1.1)
        for i in range(len(x)):
            ax.text(
                x[i],
                durations[i],
                f"{durations[0] / durations[i]:.2f}x",
                ha="center",
                va="bottom",
            )

        fig.savefig(output_dir / file)
        plt.close(fig)

        # Keyed by the method's heading, so the page groups the charts under it
        # and a backend without methods keys every chart under one empty
        # heading the template leaves out.
        dim_transform_groups[param.ndim()][transform][
            gpu_method_heading(method)
        ].append(
            {
                "path": f"pics/{file}",
                "params": case_string(param, method),
            }
        )

    facts = (
        gpu_facts(builds_root, args.cmake_cache_from)
        if gpu
        else cpu_facts(builds_root, args.cmake_cache_from)
    )
    env = Environment(
        loader=FileSystemLoader(template_path.parent), undefined=StrictUndefined
    )
    template = env.get_template(template_path.name)
    rendered = template.render(
        heading=heading,
        facts=facts,
        dim_transform_groups=dim_transform_groups,
    )
    (output_dir / f"performance_{args.backend}.rst").write_text(
        rendered, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
