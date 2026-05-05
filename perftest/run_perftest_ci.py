#!/usr/bin/env python3
import argparse
import os
import io
import re
import subprocess
from collections import defaultdict
from pathlib import Path
import uuid

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
    if param.threads == 1:
        args = ["-c", "0", str(perftest_bin), "--arg"] + perftest_args
        output = run_command("taskset", args)
    else:
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
        "--backend",
        default="fftw",
        help="FFT backend name.",
    )
    parser.add_argument(
        "--builds-root",
        default="./builds",
    )
    parser.add_argument(
        "--tag-list",
        default="",
        help="Comma-separated tags in preferred display order.",
    )
    parser.add_argument(
        "--page-template",
        default="docs/performance_backend.rst.j2",
        help="Path to the docs template page to render.",
    )
    parser.add_argument(
        "--outputs",
        default="./outputs",
        help="Output directory for generated performance report.",
    )
    args = parser.parse_args()

    builds_root = Path(args.builds_root)
    assert builds_root.exists()
    output_dir = Path(args.outputs)
    assert output_dir.exists()
    template_path = Path(args.page_template)
    assert template_path.exists()

    tags = args.tag_list.split(" ")

    helpmsg = subprocess.run(
        [builds_root / "master" / "perftest" / "perftest", "--debug=2"],
        capture_output=True,
        text=True,
    ).stdout
    ncores = int(re.search(r"opts.nthreads=(\d+)", helpmsg).groups()[0])
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

            file = f"perftestci_{uuid.uuid4()}.png"
            durations = np.array(makeplan) + np.array(setpts) + np.array(execute)
            ax.set_ylim(top=np.max(durations) * 1.1)
            for i in range(len(x)):
                ax.text(
                    x[i],
                    durations[i],
                    f"{durations[i] / durations[0]:.2f}x",
                    ha="center",
                    va="bottom",
                )

            fig.savefig(output_dir / file)
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
        backend=args.backend.upper(),
        cpu_name=cpu_info["brand_raw"],
        arch=cpu_info["arch"],
        core_count=ncores,
        flags=", ".join(cpu_info["flags"]),
        compiler_version=compiler_version,
        compiler_flags=compiler_flags,
        dim_transform_groups=dim_transform_groups,
    )
    (output_dir / f"performance_{args.backend}.rst").write_text(
        rendered, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
