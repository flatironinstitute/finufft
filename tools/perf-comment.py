"""Build the perftest comment comparing a PR head against master.

Runs both libraries' benchmarks - ``perftest`` for the CPU library through
``perftest/perftest_pr_head.py``, ``cuperftest`` for the GPU one here -
and writes one markdown body plus one plot each.

The image URLs are not known yet at this point: they name the commit that
will carry the plots, which cannot exist until the plots do. The body
carries ``CPU_IMAGE_URL`` and ``GPU_IMAGE_URL`` for the caller to
substitute once it has published them.
"""

# /// script
# dependencies = ["matplotlib", "pandas", "numpy", "py-cpuinfo", "archspec"]
# ///

# The cpu mode imports the harness beside this script, so the list above is the
# union of what both import.

import argparse
import subprocess
import sys
from pathlib import Path

# Before anything imports pyplot, since the perftest modules below do.
import matplotlib

matplotlib.use("Agg")

# Both halves measure the same cases the same way, so the case list, the
# estimator and the table have one home: the config the CPU harness carries.
# The GPU harness beside it is shared with the docs page.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perftest"))
from cuperftest_helpers import (  # noqa: E402
    gpu_total,
    gpu_args,
    gpu_cases,
    gpu_label,
    nvcc_version,
    query_gpu,
    run_cuperftest,
)
from perftest_config import (  # noqa: E402
    PROVENANCE,
    ROUNDS,
    TRANSFORMS,
    case_table,
    reduce_rounds,
    stacked_grid,
)
from perftest_pr_head import cpu_compare  # noqa: E402


def compare(master_bin: str, pr_bin: str, case: list[str]) -> list[tuple]:
    """Time both binaries on one case and return one (master, pr) pair per round.

    The two binaries are interleaved and their order alternates: whichever runs
    first pays any residual first-run cost, so a fixed order biases the ratio.
    Rounds come back in run order; nothing here reduces them.
    """
    rounds = []
    for r in range(ROUNDS):
        if r % 2:
            master = run_cuperftest(master_bin, case)
            prhead = run_cuperftest(pr_bin, case)
        else:
            prhead = run_cuperftest(pr_bin, case)
            master = run_cuperftest(master_bin, case)
        rounds.append((master, prhead))
    return rounds


def master_sha(binary: str) -> str:
    """The master commit a binary was built from, read from its own clone.

    Each pod clones master separately, so a merge landing between the two pods
    would compare the halves against different baselines. Naming the commit each
    half used makes that visible instead of silent. Not fatal: losing the label
    is not worth discarding a finished benchmark.
    """
    out = subprocess.run(
        ["git", "-C", str(Path(binary).parent), "rev-parse", "--short", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return out.stdout.strip() or "?"


def cpu_section(cpu: dict) -> list[str]:
    """The CPU half of the comment. Its plot URL is filled in at post time."""
    return [
        "## CPU",
        "",
        "_A band that brackets 1.00 resolved nothing; read the table, not the "
        "point estimate._",
        "",
        (
            f"FFT backends: `{cpu['backends']}`. `{cpu['cpu_name']}`, "
            f"{cpu['ncores']} usable processors, "
            f"{cpu['ncores_phys']} physical cores, "
            f"`{cpu['level']}`. "
            f"Baseline master `{cpu['master_sha']}`."
        ),
        "",
        "<details><summary>perftest plot</summary>",
        "",
        "![FINUFFT perftest plot](CPU_IMAGE_URL)",
        "",
        "</details>",
        "",
        "<details><summary>how the benchmarks are measured</summary>",
        "",
        PROVENANCE,
        "",
        "</details>",
        "",
        "<details><summary>per-case timings</summary>",
        "",
        *case_table(cpu["rows"]),
        "",
        "</details>",
        "",
        "<details><summary>microarchitecture and compiler</summary>",
        "",
        f"Microarchitecture: `{cpu['uarch']}`",
        "",
        f"Compiler: `{cpu['compiler_version']}`",
        "",
        f"Flags: `{cpu['compiler_flags']}`",
        "",
        "</details>",
        "",
        "<details><summary>perftest commands</summary>",
        "",
        "```",
        cpu["commands"],
        "```",
        "",
        "</details>",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    # Each half writes its own section of the comment, so the two run in
    # parallel pods and the pipeline concatenates what they produce.
    c = sub.add_parser("cpu")
    c.add_argument("--master-perftest", action="append", metavar="FFT=PATH")
    c.add_argument("--pr-perftest", action="append", metavar="FFT=PATH")
    c.add_argument("--plot-output", type=Path, required=True)
    c.add_argument("--section-output", type=Path, required=True)

    g = sub.add_parser("gpu")
    g.add_argument("--master-cuperftest", required=True)
    g.add_argument("--pr-cuperftest", required=True)
    g.add_argument("--plot-output", type=Path, required=True)
    g.add_argument("--section-output", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "cpu":
        meta = cpu_compare(args.master_perftest, args.pr_perftest, args.plot_output)
        # Every backend is built from the same clone, so the first names them all.
        meta["master_sha"] = master_sha(args.master_perftest[0].partition("=")[2])
        args.section_output.write_text("\n".join(cpu_section(meta)) + "\n")
        return

    cases = gpu_cases()
    total = len(cases) * len(TRANSFORMS)
    rows, commands = [], []
    for param in cases:
        for transform in TRANSFORMS:
            case = gpu_args(param, transform)
            label = gpu_label(param, transform)
            print(
                f"[{len(rows) + 1}/{total}] {label.replace(chr(10), ' ')}",
                file=sys.stderr,
                flush=True,
            )
            commands.append(" ".join([args.master_cuperftest] + case))
            rounds = compare(args.master_cuperftest, args.pr_cuperftest, case)
            master, prhead, low, high = reduce_rounds(rounds, gpu_total)
            rows.append((label, master, prhead, low, high))

    stacked_grid(
        [(label, master, prhead) for label, master, prhead, _, _ in rows],
        "cuFINUFFT performance change between master and latest pr HEAD",
        args.plot_output,
    )

    gpu = query_gpu()
    nvcc = nvcc_version()

    body = [
        "## GPU",
        "",
        f"`{gpu}`. Baseline master `{master_sha(args.master_cuperftest)}`.",
        "",
        "<details><summary>cuperftest plot</summary>",
        "",
        "![cuFINUFFT perftest plot](GPU_IMAGE_URL)",
        "",
        "</details>",
        "",
        "<details><summary>how the benchmarks are measured</summary>",
        "",
        PROVENANCE,
        "",
        "</details>",
        "",
        "<details><summary>per-case timings</summary>",
        "",
        *case_table(
            [
                (name, gpu_total(m), gpu_total(p), low, high)
                for name, m, p, low, high in rows
            ]
        ),
        "",
        "</details>",
        "",
        "<details><summary>device and toolkit</summary>",
        "",
        f"Device: `{gpu}`",
        "",
        "```",
        nvcc,
        "```",
        "",
        "</details>",
        "",
        "<details><summary>cuperftest commands</summary>",
        "",
        "```",
        "\n".join(commands),
        "```",
        "",
        "</details>",
    ]
    args.section_output.write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    main()
