#!/usr/bin/env python3
import argparse
import os
import io
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

from perftest_config import PARAM_LIST, NRUNS, TRANSFORMS


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


def detect_cpu_info() -> tuple[str, str]:
    cpu_name = "unknown"
    cpu_flags = ""
    with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
            elif line.startswith("flags"):
                cpu_flags = line.strip()
    return cpu_name, cpu_flags


def resolve_perftest_bin(build_dir: Path) -> Path | None:
    candidates = [
        build_dir / "perftest" / "perftest",
        build_dir / "perftest",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def discover_tags(
    builds_root: Path, versions_input: str, current_tag: str
) -> list[str]:
    preferred: list[str] = []
    if versions_input.strip():
        preferred.extend(
            tag.strip() for tag in versions_input.split(",") if tag.strip()
        )
    if current_tag and current_tag not in preferred:
        preferred.append(current_tag)

    if preferred:
        return preferred

    tags = []
    for path in sorted(builds_root.iterdir()):
        if path.is_dir():
            tags.append(path.name)
    return tags


def run_benchmarks_for_bin(perftest_bin: Path) -> pd.DataFrame:
    output_rows: list[dict[str, Number | str]] = []
    run_count = 0
    extra_args = [
        f"n_runs={NRUNS}",
        "sort=1",
        "upsampfact=0",
        "kerevalmethod=1",
        "debug=0",
        "bandwidth=1.0",
    ]

    for param in PARAM_LIST:
        param_id = param.get_hash()
        for transform in TRANSFORMS:
            perftest_args = param.args() + extra_args + [f"type={transform}"]
            output = run_command(str(perftest_bin), perftest_args)
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
            run_count += 1

    print(f"Ran {run_count} perftest invocations successfully for {perftest_bin}.")
    return pd.DataFrame(output_rows)


def render_plots_and_report(
    all_results: pd.DataFrame,
    tags: list[str],
    plot_output_dir: Path,
    template_path: Path,
    cpu_name: str,
    cpu_flags: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from jinja2 import Environment, FileSystemLoader

    plot_output_dir.mkdir(parents=True, exist_ok=True)
    if all_results.empty:
        raise RuntimeError("No performance measurements were produced")

    df = all_results

    dim_transform_groups = defaultdict(lambda: defaultdict(list))
    for param in PARAM_LIST:
        param_df = df[df["params_hash"] == param.get_hash()]
        for transform in TRANSFORMS:
            transform_df = param_df[param_df["transform"] == transform]
            transform_df = transform_df.set_index("tag")
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

    env = Environment(loader=FileSystemLoader(template_path.parent))
    template = env.get_template(template_path.name)
    cpu_flags = cpu_flags.split()
    isa_features = [flag for flag in cpu_flags if flag.startswith("avx")]
    fma_supported = "yes" if "fma" in cpu_flags else "no"
    if not isa_features:
        isa_features_text = "none detected"
    else:
        isa_features_text = ", ".join(isa_features)
    rendered = template.render(
        cpu_name=cpu_name,
        simd_features=isa_features_text,
        fma_supported=fma_supported,
        dim_transform_groups=dim_transform_groups,
    )
    template_path.with_suffix("").write_text(rendered, encoding="utf-8")


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
        default="docs/perftest_timings.rst.j2",
        help="Path to the docs template page to render.",
    )
    args = parser.parse_args()

    builds_root = Path(args.builds_root)
    if not builds_root.exists():
        raise RuntimeError(f"Build root does not exist: {builds_root}")

    tags = discover_tags(builds_root, args.tag_list, "")

    all_frames: list[pd.DataFrame] = []
    used_tags: list[str] = []
    for tag in tags:
        build_dir = builds_root / tag
        perftest_bin = resolve_perftest_bin(build_dir)
        if perftest_bin is None:
            print(f"Skipping {tag}: could not find executable perftest in {build_dir}")
            continue
        tag_df = run_benchmarks_for_bin(perftest_bin)
        if tag_df.empty:
            print(f"Skipping {tag}: benchmark produced no rows")
            continue
        tag_df["tag"] = tag
        all_frames.append(tag_df)
        used_tags.append(tag)

    if not all_frames:
        raise RuntimeError("No benchmark data collected from any build directory")

    all_results = pd.concat(all_frames, ignore_index=True)

    cpu_name, cpu_flags = detect_cpu_info()

    render_plots_and_report(
        all_results=all_results,
        tags=used_tags,
        plot_output_dir=Path(args.plot_output_dir),
        template_path=Path(args.docs_page_path),
        cpu_name=cpu_name,
        cpu_flags=cpu_flags,
    )


if __name__ == "__main__":
    main()
