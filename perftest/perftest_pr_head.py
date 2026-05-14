"""Compare perftest results between master and the PR-head build.

Outputs ``GITHUB_OUTPUT`` key/value lines on stdout (cpu_name, arch,
ncores, cpu_flags, compiler_version, compiler_flags, commands) and
writes a comparison plot SVG to ``--plot-output``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perftest_config import PARAM_LIST, TRANSFORMS
from perftest_helpers import (
    METRIC_COLUMN,
    build_command,
    cpu_metadata,
    read_cmake_metadata,
    read_ncores,
    run_perftest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-perftest", default="../builds/master/perftest/perftest"
    )
    parser.add_argument("--pr-perftest", default="../builds/pr-head/perftest/perftest")
    parser.add_argument("--plot-output", default="figure.png")
    args = parser.parse_args()

    cpu = cpu_metadata()
    print(f"cpu_name={cpu['cpu_name']}")
    print(f"arch={cpu['arch']}")
    print(f"cpu_flags={cpu['flags']}")
    print(f"ncores={read_ncores(Path(args.master_perftest))}")

    meta = read_cmake_metadata(Path(args.master_perftest).parent.parent)
    print(f"compiler_version={meta['compiler_version']}")
    print(f"compiler_flags={meta['compiler_flags']}")

    fig, axs = plt.subplots(
        len(PARAM_LIST),
        len(TRANSFORMS),
        figsize=(len(TRANSFORMS) * 4, len(PARAM_LIST) * 4),
        squeeze=False,
    )
    stages = ["execute", "setpts", "makeplan"]
    cmd_list: list[str] = []
    total = len(PARAM_LIST) * len(TRANSFORMS)
    done = 0
    for i, param in enumerate(PARAM_LIST):
        for j, transform in enumerate(TRANSFORMS):
            done += 1
            t0 = time.monotonic()
            print(
                f"[{done}/{total}] type={transform} "
                f"{param.pretty_string().replace(chr(10), ' ')}",
                file=sys.stderr,
                flush=True,
            )
            master_cmd = build_command(param, transform, args.master_perftest)
            cmd_list.append(" ".join(master_cmd))
            master_df = run_perftest(master_cmd)
            prhead_df = run_perftest(build_command(param, transform, args.pr_perftest))
            print(
                f"  done in {time.monotonic() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )

            ax = axs[i][j]
            bottoms = np.array([0.0, 0.0])
            barc = None
            for stage in stages:
                heights = (
                    master_df.loc[stage, METRIC_COLUMN],
                    prhead_df.loc[stage, METRIC_COLUMN],
                )
                barc = ax.bar(
                    ["master", " pr head"], heights, bottom=bottoms, label=stage
                )
                bottoms += heights
            ax.set_ylim(top=np.max(bottoms) * 1.1)
            ax.bar_label(barc, labels=["1.00x", f"{bottoms[0] / bottoms[1]:.2f}x"])
            ax.set_ylabel("time (ms)")
            ax.set_title(f"type:{transform} {param.pretty_string()}")
    axs[0][0].legend()
    fig.suptitle("Performance change between master and latest pr HEAD", fontsize=24)
    fig.tight_layout(pad=2, h_pad=2)
    fig.savefig(args.plot_output, dpi=150)
    plt.close(fig)

    # Avoid f-string with backslash (PEP 701 only, not portable to <3.12).
    newline = "\n"
    print(f"commands={json.dumps(newline.join(cmd_list))}")


if __name__ == "__main__":
    main()
