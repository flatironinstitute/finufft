import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import io
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cpuinfo import get_cpu_info

from perftest_config import Params, PARAM_LIST, NRUNS, TRANSFORMS


def get_command(param: Params, transform: int, binary_path: str) -> list[str]:
    extra_args = [
        f"--n_runs={NRUNS}",
        "--sort=1",
        "--upsampfact=0",
        "--kerevalmethod=1",
        "--debug=0",
        "--bandwidth=1.0",
    ]
    return [binary_path] + param.args() + extra_args + [f"--type={transform}"]


def run_one_perftest(cmd: str) -> pd.DataFrame:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return pd.read_csv(
        io.StringIO(
            "\n".join(
                [
                    line
                    for line in result.stdout.splitlines()
                    if not line.startswith("#")
                ]
            )
        ),
        sep=",",
    ).set_index("event")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-perftest", default="../builds/master/perftest/perftest"
    )
    parser.add_argument("--pr-perftest", default="../builds/pr-head/perftest/perftest")
    parser.add_argument("--plot-output", default="figure.png")

    args = parser.parse_args()

    cpu_info = get_cpu_info()
    simd_features = ", ".join([flag for flag in cpu_info["flags"] if "avx" in flag])
    fma_supported = "yes" if "fma" in cpu_info["flags"] else "no"

    print(f"cpu_name={cpu_info['brand_raw']}")
    print(f"simd_features={simd_features}")
    print(f"fma_supported={fma_supported}")

    compiler_version = "NA"
    compiler_flags = "NA"
    with open(Path(args.master_perftest).parent.parent / "CMakeCache.txt", "r") as f:
        for line in f:
            if "CMAKE_CXX_COMPILER:FILEPATH=" in line:
                cxx = line.removeprefix("CMAKE_CXX_COMPILER:FILEPATH=")
                compiler_version = subprocess.run(
                    [cxx.strip(), "--version"], capture_output=True, text=True
                ).stdout.split("\n")[0]
            if "FINUFFT_ARCH_FLAGS:STRING=" in line:
                compiler_flags = line.removeprefix("FINUFFT_ARCH_FLAGS:STRING=").strip()
    print(f"compiler_version={compiler_version}")
    print(f"compiler_flags={compiler_flags}")

    fig, axs = plt.subplots(
        len(PARAM_LIST),
        len(TRANSFORMS),
        figsize=(len(TRANSFORMS) * 4, len(PARAM_LIST) * 4),
        squeeze=False,
    )
    stages = ["execute", "setpts", "makeplan"]
    cmd_list = []
    for i, param in enumerate(PARAM_LIST):
        for j, transform in enumerate(TRANSFORMS):
            master_cmd = get_command(param, transform, args.master_perftest)
            cmd_list.append(" ".join(master_cmd))
            master_df = run_one_perftest(master_cmd)
            prhead_df = run_one_perftest(
                get_command(param, transform, args.pr_perftest)
            )
            ax = axs[i][j]
            bottoms = np.array([0, 0], dtype=np.float64)
            barc = None
            for stage in stages:
                heights = (
                    master_df.loc[stage, "mean(ms)"],
                    prhead_df.loc[stage, "mean(ms)"],
                )
                barc = ax.bar(
                    ["master", " pr head"], heights, bottom=bottoms, label=stage
                )
                bottoms += heights
            ax.set_ylim(top=np.max(bottoms) * 1.1)
            ax.bar_label(barc, labels=["1.00x", f"{bottoms[1] / bottoms[0]:.2f}x"])
            ax.set_ylabel("time (ms)")
            ax.set_title(f"type:{transform} {param.pretty_string()}")
    axs[0][0].legend()
    fig.suptitle("Performance change between master and latest pr HEAD", fontsize=24)
    fig.tight_layout(pad=2, h_pad=2)
    fig.savefig(args.plot_output, dpi=150)
    plt.close(fig)

    print(f"commands={json.dumps('\n'.join(cmd_list))}")


if __name__ == "__main__":
    main()
