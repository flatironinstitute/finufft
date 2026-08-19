"""Shared helpers for the perftest CI scripts.

These utilities are imported by both ``perftest_pr_head.py`` (the
PR-vs-master comparison) and ``run_perftest_ci.py`` (the multi-tag
matrix). Keeping them here avoids drift between the two scripts.

The reported timing metric is ``METRIC_COLUMN``, which ``perftest_config``
holds because both harnesses emit it: ``perftest.cpp`` and ``cuperftest.cu``
give aggregate statistics rather than per-run rows, and the fastest run is the
one least polluted by whatever else the machine was doing.
"""

from __future__ import annotations

import ctypes
import functools
import io
import os
import platform
from dataclasses import fields as dataclass_fields
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable

import archspec.cpu
import pandas as pd
from cpuinfo import get_cpu_info

from perftest_config import METRIC_COLUMN, NRUNS, Params, cpu_args

# finufft's own defaults: 2 is its spread_sort heuristic and 0 its auto
# upsampling, so the bot measures what a caller gets. The thread count is the
# exception, because a case is defined by the thread count it runs at.
EXTRA_ARGS: list[str] = [
    f"--n_runs={NRUNS}",
    "--sort=2",
    "--upsampfact=0",
    "--debug=0",
]

MPOL_INTERLEAVE = 3
# set_mempolicy has no glibc wrapper, and libnuma is not in the CI image, so it
# goes through syscall(). The number is per architecture.
SYS_SET_MEMPOLICY = {"x86_64": 238, "aarch64": 237}.get(platform.machine())


def interleave_memory() -> None:
    """Spread the calling process's pages over every NUMA node.

    What ``numactl --interleave=all`` does, as the syscall it wraps, because the
    CI image carries no numactl. A memory policy survives ``execve``, so setting
    it in the forked child covers the binary that replaces it.

    Failure is silent, and the catch is broad because this runs as a
    ``preexec_fn``: an exception there aborts the measurement, and measuring
    first touch beats measuring nothing.
    """
    try:
        online = Path("/sys/devices/system/node/online").read_text().strip()
        nodes = 0
        for span in online.split(","):
            lo, _, hi = span.partition("-")
            for node in range(int(lo), int(hi or lo) + 1):
                nodes |= 1 << node
        mask = ctypes.c_ulong(nodes)
        ctypes.CDLL(None, use_errno=True).syscall(
            SYS_SET_MEMPOLICY,
            MPOL_INTERLEAVE,
            ctypes.byref(mask),
            8 * ctypes.sizeof(mask),
        )
    except Exception:
        pass


def build_command(
    param: Params, transform: int, binary_path: str, cpu: int | None = None
) -> list[str]:
    """Build the perftest invocation for a single (param, transform) pair.

    Single-threaded runs are pinned with ``taskset``. The CPU comes from this
    process's affinity mask: a cpuset-restricted container need not hold CPU 0,
    and taskset fails with EINVAL when asked for a CPU outside the mask.
    """
    perftest_args = (
        cpu_args(param, len(physical_cores())) if param.threads == 0 else param.args()
    )
    perftest_args += EXTRA_ARGS + [f"--type={transform}"]
    if param.threads == 1:
        if cpu is None:
            cpu = min(os.sched_getaffinity(0))
        return ["taskset", "-c", str(cpu), binary_path, "--arg"] + perftest_args
    # A run that takes every core also takes every memory controller, so
    # run_perftest interleaves its pages: first touch would put the grid on
    # whichever node the round allocated it, a floor that moves between rounds.
    return [binary_path] + perftest_args


def _sysfs(cpu: int, *parts: str) -> str:
    """One sysfs topology field of a CPU, empty where the kernel omits it."""
    try:
        return Path("/sys/devices/system/cpu", f"cpu{cpu}", *parts).read_text().strip()
    except OSError:
        return ""


def _numa_node(cpu: int) -> str:
    """The NUMA node of a CPU, which is the memory controller it is local to.

    The node rather than the package: one EPYC socket is several nodes, so the
    package would group CPUs that do not in fact share a controller.
    """
    nodes = Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node*")
    return next((n.name for n in sorted(nodes)), "") or _sysfs(
        cpu, "topology", "physical_package_id"
    )


def _last_level_cache(cpu: int) -> str:
    """The CPUs sharing this one's L3, which names the domain it sits in.

    The sibling list rather than the cache id: not every kernel exports an id,
    and the list is unique per domain on the ones that do not.
    """
    cache = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
    for index in sorted(cache.glob("index*")):
        if _sysfs(cpu, "cache", index.name, "level") == "3":
            return _sysfs(cpu, "cache", index.name, "shared_cpu_list")
    return ""


@functools.lru_cache(maxsize=None)
def physical_cores(n: int | None = None) -> list[int]:
    """``n`` CPUs of this process's affinity mask, sharing as little as possible.

    The mask is the authority on what the pod may use. Three things a pair of
    CPUs can share, in the order that hurts a benchmark most: the execution
    units of one physical core, one L3, one memory controller. The list takes
    one CPU per physical core, then walks the L3 domains round robin inside a
    NUMA node, then walks the nodes round robin, so consecutive cases land on a
    different node and a different L3 before any L3 takes a second case.

    A machine that exposes no L3 domains or one node degrades to one group,
    which still gives a core of its own to every case. ``n`` of None asks for
    every core, which is the count a ``--threads=0`` case runs with.
    """
    by_node: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        # The sibling list names the physical core, so an SMT sibling of a core
        # already taken is skipped whatever the kernel calls the core itself.
        core = _sysfs(cpu, "topology", "thread_siblings_list") or str(cpu)
        if core in seen:
            continue
        seen.add(core)
        by_node[_numa_node(cpu)][_last_level_cache(cpu)].append(cpu)

    def interleave(groups) -> list[int]:
        return [cpu for row in zip_longest(*groups) for cpu in row if cpu is not None]

    cores = interleave([interleave(domains.values()) for domains in by_node.values()])
    if n is None:
        return cores
    assert len(cores) >= n, f"{n} cores wanted, {len(cores)} in the affinity mask"
    return cores[:n]


def measure_cases(
    params: list[Params],
    run_one: Callable[[int, int | None], Any],
    concurrent: bool = True,
):
    """Measure every case through ``run_one(index, cpu)``, in the input order.

    Single-threaded cases run concurrently, one core per param row: a row's
    three transforms ride the same core and L3 in turn. Whole-node cases run
    alone after them, under an interleaved memory policy (a ``preexec_fn`` is
    only safe once the pool has joined).

    Pass ``concurrent=False`` for a device only one case can hold at a time.
    """
    if not concurrent:
        return [run_one(i, None) for i in range(len(params))]
    rows: dict[tuple, list[int]] = {}
    for i, param in enumerate(params):
        if param.threads == 1:
            key = tuple(getattr(param, f.name) for f in dataclass_fields(param))
            rows.setdefault(key, []).append(i)
    results: dict[int, Any] = {}
    if rows:
        cores = physical_cores(len(rows))

        def run_row(row_cases: list[int], row: int) -> None:
            for i in row_cases:
                results[i] = run_one(i, cores[row])

        with ThreadPoolExecutor(max_workers=len(rows)) as pool:
            futures = [
                pool.submit(run_row, row_cases, r)
                for r, row_cases in enumerate(rows.values())
            ]
            for future in futures:
                future.result()
    for i, param in enumerate(params):
        if param.threads != 1:
            results[i] = run_one(i, None)
    return [results[i] for i in range(len(params))]


def run_perftest(cmd: list[str], interleave: bool = False) -> pd.DataFrame:
    """Run a perftest command and parse its CSV output.

    ``interleave`` spreads the run's pages over every NUMA node, which is right
    for a whole-node case and wrong for a pinned one: a case confined to one
    core wants its grid on that core's node.

    Raises RuntimeError (with stderr surfaced) on non-zero exit.
    """
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=interleave_memory if interleave else None,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"perftest invocation failed ({' '.join(cmd)}):\n{exc.stderr}"
        ) from exc

    csv_text = "\n".join(
        line for line in result.stdout.splitlines() if not line.startswith("#")
    )
    return pd.read_csv(io.StringIO(csv_text), sep=",").set_index("event")


def usable_ncores() -> int:
    """Cores this process may run on, which is what the timings reflect.

    A cpuset-restricted pod sees every core on the node, so a total core count
    overstates the machine the benchmark actually got.
    """
    return len(os.sched_getaffinity(0))


def read_cmake_metadata(build_dir: Path) -> dict[str, str]:
    """Extract compiler version and arch flags from a CMakeCache.txt."""
    compiler_version = "NA"
    compiler_flags = "NA"
    cache = Path(build_dir) / "CMakeCache.txt"
    with open(cache, "r") as f:
        for line in f:
            if "CMAKE_CXX_COMPILER:FILEPATH=" in line:
                cxx = line.removeprefix("CMAKE_CXX_COMPILER:FILEPATH=").strip()
                compiler_version = subprocess.run(
                    [cxx, "--version"], capture_output=True, text=True
                ).stdout.split("\n")[0]
            elif "FINUFFT_ARCH_FLAGS:STRING=" in line:
                compiler_flags = line.removeprefix("FINUFFT_ARCH_FLAGS:STRING=").strip()
    return {"compiler_version": compiler_version, "compiler_flags": compiler_flags}


def cpu_metadata() -> dict[str, Any]:
    """Wrap ``py-cpuinfo`` into the dict shape used by both scripts."""
    info = get_cpu_info()
    host = archspec.cpu.host()
    return {
        "cpu_name": info["brand_raw"],
        "arch": info["arch"],
        # archspec names the psABI level, e.g. x86_64_v4, from its own flag
        # tables; deriving it from raw flag names gets vendor spellings wrong.
        "uarch": host.name,
        "level": host.generic.name.replace("_", "-"),
    }
