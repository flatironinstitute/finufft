#!/usr/bin/env python3
import argparse
import csv
import json
import os
import io
import subprocess
import hashlib
from collections import defaultdict
from dataclasses import dataclass, fields, asdict
from numbers import Number
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
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

    def get_hash(self) -> str:
        return hashlib.md5(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def args(self) -> list[str]:
        return [f"{f.name}={getattr(self, f.name)}" for f in fields(self)]

    def pretty_string(self) -> str:
        return ", ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


NRUNS = 10

PARAM_LIST = [
    Params("f", 1e4, 1, 1, 1, 1, 1e7, 1e-4),
    Params("d", 1e4, 1, 1, 1, 1, 1e7, 1e-9),
    Params("f", 320, 320, 1, 1, 1, 1e7, 1e-5),
    Params("d", 320, 320, 1, 1, 1, 1e7, 1e-9),
    Params("f", 320, 320, 1, 1, 0, 1e7, 1e-5),
    Params("d", 192, 192, 128, 1, 0, 1e7, 1e-7),
]

TRANSFORMS = [3, 2, 1]
DEFAULT_EXTRA_ARGS = [
    f"n_runs={NRUNS}",
    "sort=1",
    "upsampfact=0",
    "kerevalmethod=1",
    "debug=0",
    "bandwidth=1.0",
]


def set_arg(args: list[str], key: str, value: str) -> list[str]:
    key_prefix = f"{key}="
    updated = args.copy()
    for i, arg in enumerate(updated):
        if arg.startswith(key_prefix):
            updated[i] = f"{key}={value}"
            return updated
    updated.append(f"{key}={value}")
    return updated


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


def gather_results(args) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from jinja2 import Environment, FileSystemLoader

    matrix = json.loads(args.matrix_json)
    tags = [entry["tag"] for entry in matrix.get("include", [])]
    base_dir = Path(args.input_dir)
    plot_output_dir = Path(args.plot_output_dir)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    generated_images: list[dict[str, str]] = []

    for tag in tags:
        try:
            frame = pd.read_csv(base_dir / f"{tag}.csv")
        except pd.errors.EmptyDataError:
            print(f"Skipped version {tag}")
            continue
        frame["tag"] = tag
        frames.append(frame)

    if not frames:
        raise RuntimeError("Matrix did not produce any performance measurements")

    df = pd.concat(frames, ignore_index=True)

    dim_transform_groups = defaultdict(lambda: defaultdict(list))
    for param in PARAM_LIST:
        param_df = df[df["params_hash"] == param.get_hash()]
        for transform in TRANSFORMS:
            transform_df = param_df[param_df["transform"] == transform]
            transform_df.set_index("tag", inplace=True)
            xs = []
            makeplan = []
            setpts = []
            execute = []
            makeplan_std = []
            setpts_std = []
            execute_std = []
            missing = []
            for i, tag in enumerate(tags):
                try:
                    perftest = transform_df.loc[tag]
                    makeplan.append(perftest["makeplan_mean(ms)"])
                    setpts.append(perftest["setpts_mean(ms)"])
                    execute.append(perftest["execute_mean(ms)"])
                    makeplan_std.append(perftest["makeplan_std(ms)"])
                    setpts_std.append(perftest["setpts_std(ms)"])
                    execute_std.append(perftest["execute_std(ms)"])
                    xs.append(i)
                except KeyError:
                    missing.append(i)
            fig, ax = plt.subplots()
            y = np.array(execute)
            std = np.array(execute_std)
            (line,) = ax.plot(xs, execute, "o-", label="execute")
            ax.fill_between(
                xs,
                np.maximum(y - std, 0),
                (y + std),
                color=line.get_color(),
                alpha=0.2,
            )

            y = np.array(makeplan)
            std = np.array(makeplan_std)
            (line,) = ax.plot(xs, makeplan, "o-", label="makeplan")
            ax.fill_between(
                xs,
                np.maximum(y - std, 0),
                (y + std),
                color=line.get_color(),
                alpha=0.2,
            )

            y = np.array(setpts)
            std = np.array(setpts_std)
            (line,) = ax.plot(xs, setpts, "o-", label="setpts")
            ax.fill_between(
                xs,
                np.maximum(y - std, 0),
                (y + std),
                color=line.get_color(),
                alpha=0.2,
            )

            all_tags = dict(enumerate(tags))
            ax.set_xticks(list(all_tags.keys()))
            ax.set_xticklabels(list(all_tags.values()))
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Version")
            ax.set_ylabel("Mean time (ms) ± std")

            ax.legend()
            image_path = (
                plot_output_dir / f"perftestci_{param.get_hash()}_type_{transform}.png"
            )
            fig.savefig(image_path)
            dim_transform_groups[param.ndim()][transform].append(
                {
                    "path": f"pics/{image_path.name}",
                    "params": f"{param.pretty_string()}",
                }
            )
            print(param.pretty_string())
            plt.close(fig)

    template_path = Path(args.template_path)
    env = Environment(loader=FileSystemLoader(template_path.parent))
    template = env.get_template(template_path.name)
    cpu_flags = args.cpu_flags.partition(":")[-1].strip().split()
    isa_features = [flag for flag in cpu_flags if flag.startswith("avx")]
    fma_supported = "yes" if "fma" in cpu_flags else "no"
    if not isa_features:
        isa_features_text = "none detected"
    else:
        isa_features_text = ", ".join(isa_features)
    rendered = template.render(
        cpu_name=args.cpu_name.partition(":")[-1],
        simd_features=isa_features_text,
        fma_supported=fma_supported,
        dim_transform_groups=dim_transform_groups,
    )
    template_path.with_suffix("").write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run perftests or gather per-tag perftest CSV results."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run perftest matrix and write per-tag CSV."
    )
    run_parser.add_argument(
        "--perftest-bin",
        default="./build/perftest/perftest",
        help="Path to perftest executable.",
    )
    run_parser.add_argument(
        "--output-csv",
        default="matrix-results/perftest_ci_results.csv",
        help="Output CSV path for collected perftest rows.",
    )
    gather_parser = subparsers.add_parser(
        "gather", help="Gather per-tag CSV files into one CSV."
    )
    gather_parser.add_argument(
        "--matrix-json",
        required=True,
        help='Matrix JSON like {"include":[{"tag":"...","commit":"..."}]}.',
    )
    gather_parser.add_argument(
        "--input-dir",
        default="matrix-results",
        help="Input directory containing per-tag CSV files (gather mode).",
    )
    gather_parser.add_argument(
        "--plot-output-dir",
        default="docs/pics",
        help="Output directory for generated performance plot images.",
    )
    gather_parser.add_argument(
        "--template-path",
        default="docs/perftest_timings.rst.j2",
        help="Path to docs page template.",
    )
    gather_parser.add_argument("--cpu-name", help="Machine cpu name")
    gather_parser.add_argument("--cpu-flags", default="", help="Machine cpu flags line")
    args = parser.parse_args()

    if args.mode == "gather":
        gather_results(args)

        return
    run_count = 0
    output_rows = []

    for param in PARAM_LIST:
        param_id = param.get_hash()
        for transform in TRANSFORMS:
            perftest_args = param.args() + DEFAULT_EXTRA_ARGS + [f"type={transform}"]
            output = run_command(args.perftest_bin, perftest_args)
            test_res = pd.read_csv(
                io.StringIO(
                    "\n".join(
                        [
                            line
                            for line in output.splitlines()
                            if not line.startswith("#")
                        ]
                    )
                ),
                sep=",",
            )
            test_res = test_res.set_index("event")
            output_rows.append(
                {
                    "params_hash": param_id,
                    "transform": transform,
                    "makeplan_mean(ms)": test_res.at["makeplan", "mean(ms)"],
                    "makeplan_std(ms)": test_res.at["makeplan", "std(ms)"],
                    "setpts_mean(ms)": test_res.at["setpts", "mean(ms)"],
                    "setpts_std(ms)": test_res.at["setpts", "std(ms)"],
                    "execute_mean(ms)": test_res.at["execute", "mean(ms)"],
                    "execute_std(ms)": test_res.at["execute", "std(ms)"],
                }
            )
    pd.DataFrame(output_rows).to_csv(args.output_csv, index=False)
    print(f"Ran {run_count} perftest invocations successfully.")
    print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
