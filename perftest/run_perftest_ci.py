#!/usr/bin/env python3
"""Run the perftest matrix across tagged builds and render a docs page."""

import argparse
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader

from perftest_config import PARAM_LIST, TRANSFORMS
from perftest_helpers import (
    METRIC_COLUMN,
    build_command,
    cpu_metadata,
    read_cmake_metadata,
    read_ncores,
    run_perftest,
)


def run_for_tag(param, perftest_bin: Path, transform: int):
    if not perftest_bin.exists():
        return None
    return run_perftest(build_command(param, transform, str(perftest_bin)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run perftest matrix and generate plots."
    )
    parser.add_argument("--backend", default="fftw", help="FFT backend name.")
    parser.add_argument("--builds-root", default="./builds")
    parser.add_argument(
        "--tag-list",
        default="",
        help="Space-separated tags in preferred display order.",
    )
    parser.add_argument(
        "--page-template",
        default="docs/performance_backend.rst.j2",
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

    ncores = read_ncores(builds_root / "master" / "perftest" / "perftest")
    meta = read_cmake_metadata(builds_root / args.cmake_cache_from)

    dim_transform_groups = defaultdict(lambda: defaultdict(list))

    total = len(PARAM_LIST) * len(TRANSFORMS)
    done = 0
    for param in PARAM_LIST:
        for transform in TRANSFORMS:
            done += 1
            t0 = time.monotonic()
            print(
                f"[{done}/{total}] type={transform} "
                f"{param.pretty_string().replace(chr(10), ' ')}",
                file=sys.stderr,
                flush=True,
            )
            x: list[str] = []
            setpts: list[float] = []
            makeplan: list[float] = []
            execute: list[float] = []
            for tag in tags:
                tag_df = run_for_tag(
                    param, builds_root / tag / "perftest" / "perftest", transform
                )
                if tag_df is None or tag_df.empty:
                    continue
                print(f"  tag={tag} ok", file=sys.stderr, flush=True)
                x.append(tag)
                makeplan.append(tag_df.loc["makeplan", METRIC_COLUMN])
                setpts.append(tag_df.loc["setpts", METRIC_COLUMN])
                execute.append(tag_df.loc["execute", METRIC_COLUMN])
            print(
                f"  done in {time.monotonic() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )

            if len(x) < 2:
                continue

            fig, ax = plt.subplots()
            ax.stackplot(
                x, execute, setpts, makeplan, labels=["execute", "setpts", "makeplan"]
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Version")
            ax.set_ylabel("Min time (ms)")
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
                    "params": param.pretty_string(),
                }
            )

    cpu = cpu_metadata()
    env = Environment(loader=FileSystemLoader(template_path.parent))
    template = env.get_template(template_path.name)
    rendered = template.render(
        backend=args.backend.upper(),
        cpu_name=cpu["cpu_name"],
        arch=cpu["arch"],
        core_count=ncores,
        flags=cpu["flags"],
        compiler_version=meta["compiler_version"],
        compiler_flags=meta["compiler_flags"],
        dim_transform_groups=dim_transform_groups,
    )
    (output_dir / f"performance_{args.backend}.rst").write_text(
        rendered, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
