import argparse
import subprocess
from datetime import datetime
import io

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perftest_config import PARAM_LIST, NRUNS, TRANSFORMS


def run_one_perftest(param, transform, binary_path: str) -> pd.DataFrame:

    extra_args = [
        f"n_runs={NRUNS}",
        "sort=1",
        "upsampfact=0",
        "kerevalmethod=1",
        "debug=0",
        "bandwidth=1.0",
    ]
    cmd = [binary_path] + param.args() + extra_args + [f"type={transform}"]
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
    fig, axs = plt.subplots(
        len(PARAM_LIST),
        len(TRANSFORMS),
        figsize=(len(TRANSFORMS) * 4, len(PARAM_LIST) * 4),
        squeeze=False,
    )
    stages = ["execute", "makeplan", "setpts"]
    for i, param in enumerate(PARAM_LIST):
        for j, transform in enumerate(TRANSFORMS):
            master_df = run_one_perftest(param, transform, args.master_perftest)
            prhead_df = run_one_perftest(param, transform, args.pr_perftest)
            ax = axs[i][j]
            bottoms = np.array([0, 0], dtype=np.float64)
            for stage in stages:
                heights = (
                    master_df.loc[stage, "mean(ms)"],
                    prhead_df.loc[stage, "mean(ms)"],
                )
                ax.bar(["master", " pr head"], heights, bottom=bottoms, label=stage)
                bottoms += heights
            ax.set_ylabel("time (ms)")
            ax.set_title(f"type:{transform} {param.pretty_string()}")
    axs[0][-1].legend()
    fig.suptitle("Performance change between master and latest pr HEAD", fontsize=24)
    fig.tight_layout(pad=2, h_pad=2)
    fig.savefig(args.plot_output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
