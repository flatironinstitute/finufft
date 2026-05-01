#!/usr/bin/env python3
import argparse
import os
import io
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader
from cpuinfo import get_cpu_info

from perftest_config import Params, PARAM_LIST, NRUNS, TRANSFORMS


def run_command(command: str, args: list[str]) -> str:
    cmd = [command] + args
    print("Running command:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stderr.strip():
        print(result.stderr)
    return result.stdout


def run_benchmarks_for_bin(
    param: Params, perftest_bin: Path, transform: int
) -> pd.DataFrame:
    if not perftest_bin.exists():
        return pd.DataFrame()
    extra_args = [
        f"--n_runs={NRUNS}",
        "--sort=1",
        "--upsampfact=0",
        "--kerevalmethod=1",
        "--debug=0",
        "--bandwidth=1.0",
    ]
    perftest_args = param.args() + extra_args + [f"--type={transform}"]
    output = run_command(str(perftest_bin), perftest_args)
    return pd.read_csv(
        io.StringIO(
            "\n".join(
                [line for line in output.splitlines() if not line.startswith("#")]
            )
        ),
        sep=",",
    ).set_index("event")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run perftest matrix and generate plots."
    )
    parser.add_argument(
        "--builds-root",
        default="/builds",
    )
    parser.add_argument(
        "--tag-list",
        default="",
        help="Comma-separated tags in preferred display order.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default="docs/pics",
        help="Output directory for generated performance plot images.",
    )
    parser.add_argument(
        "--docs-page-path",
        default="docs/performance_change.rst.j2",
        help="Path to the docs template page to render.",
    )
    args = parser.parse_args()

    builds_root = Path(args.builds_root)
    assert builds_root.exists()
    plot_path = Path(args.plot_output_dir)
    assert plot_path.exists()
    template_path = Path(args.docs_page_path)
    assert template_path.exists()

    tags = args.tag_list.split(",")

    compiler_version = "NA"
    compiler_flags = "NA"
    with open(builds_root / tags[-1] / "CMakeCache.txt", "r") as f:
        for line in f:
            if "CMAKE_CXX_COMPILER:FILEPATH=" in line:
                cxx = line.removeprefix("CMAKE_CXX_COMPILER:FILEPATH=")
                compiler_version = subprocess.run(
                    [cxx.strip(), "--version"], capture_output=True, text=True
                ).stdout.split("\n")[0]
            if "FINUFFT_ARCH_FLAGS:STRING=" in line:
                compiler_flags = line.removeprefix("FINUFFT_ARCH_FLAGS:STRING=").strip()
    plot_num = 0

    dim_transform_groups = defaultdict(lambda: defaultdict(list))

    for param in PARAM_LIST:
        for transform in TRANSFORMS:
            x = []
            setpts = []
            makeplan = []
            execute = []
            for tag in tags:
                tag_df = run_benchmarks_for_bin(
                    param, builds_root / tag / "perftest" / "perftest", transform
                )
                if tag_df.empty:
                    continue
                x.append(tag)
                makeplan.append(tag_df.loc["makeplan", "mean(ms)"])
                setpts.append(tag_df.loc["setpts", "mean(ms)"])
                execute.append(tag_df.loc["execute", "mean(ms)"])

            if len(x) < 2:
                continue

            fig, ax = plt.subplots()
            ax.stackplot(
                x, execute, setpts, makeplan, labels=["execute", "setpts", "makeplan"]
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Version")
            ax.set_ylabel("Mean time (ms)")
            ax.legend()

            file = f"perftestci_{(plot_num := plot_num + 1)}.png"
            fig.savefig(plot_path / file)
            plt.close(fig)

            dim_transform_groups[param.ndim()][transform].append(
                {
                    "path": f"pics/{file}",
                    "params": f"{param.pretty_string()}",
                }
            )
    cpu_info = get_cpu_info()
    env = Environment(loader=FileSystemLoader(template_path.parent))
    template = env.get_template(template_path.name)
    rendered = template.render(
        cpu_name=cpu_info["brand_raw"],
        compiler_version=compiler_version,
        compiler_flags=compiler_flags,
        simd_features=", ".join([flag for flag in cpu_info["flags"] if "avx" in flag]),
        fma_supported="fma" in cpu_info["flags"],
        dim_transform_groups=dim_transform_groups,
    )
    template_path.with_suffix("").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
