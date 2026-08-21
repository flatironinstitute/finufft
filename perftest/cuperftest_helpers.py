"""Run ``cuperftest`` and describe its cases.

The PR comment and the docs page both time the GPU library, so the case list,
the invocation, the event names and the device description have one home here,
the way ``perftest_helpers`` holds them for the CPU library.
"""

import csv
import io
import subprocess

from perftest_config import (
    GPU_WHOLE_NODE_CORES,
    CPU_STAGES,
    METRIC_COLUMN,
    NRUNS,
    PARAM_LIST,
    pretty_number,
)

# Params fields cuperftest also understands. threads is CPU-only, so entries
# that differ only in it name one GPU case.
GPU_FIELDS = ("prec", "N1", "N2", "N3", "ntransf", "M", "tol")


def gpu_args(param, transform: int) -> list[str]:
    """cuperftest invocation for one case of the shared CPU parameter list."""
    # 1 is cufinufft's own gpu_sort, and no upsampfac argument leaves gpu_upsampfac
    # at its auto default, so the case measures the heuristics a caller gets.
    # A whole-node case's M is per core in the list; the GPU multiplies it back
    # up by the core count the case was sized for.
    shared = []
    for f in GPU_FIELDS:
        value = getattr(param, f)
        if f == "M" and param.threads == 0:
            value = int(value * GPU_WHOLE_NODE_CORES)
        shared.append(f"--{f}={pretty_number(value)}")
    return shared + [
        f"--type={transform}",
        f"--n_runs={NRUNS}",
        "--sort=1",
        "--debug=0",
    ]


def _wrap(fields: list[str]) -> str:
    """Four fields per line, as ``Params.pretty_string`` wraps them."""
    return "\n".join(" ".join(fields[i : i + 4]) for i in range(0, len(fields), 4))


def _fields(param) -> list[str]:
    return [f"{f}:{pretty_number(getattr(param, f))}" for f in GPU_FIELDS]


def gpu_params_string(param) -> str:
    """The case in the CPU page's style, minus the CPU-only thread count."""
    return _wrap(_fields(param))


def gpu_label(param, transform: int) -> str:
    """Plot title: the case, named by the transform it runs."""
    return _wrap([f"type:{transform}"] + _fields(param))


def gpu_cases() -> list:
    """The CPU parameter list with its CPU-only distinctions collapsed."""
    seen, cases = set(), []
    for param in PARAM_LIST:
        key = tuple(getattr(param, f) for f in GPU_FIELDS)
        if key not in seen:
            seen.add(key)
            cases.append(param)
    return cases


def run_cuperftest(binary: str, args: list[str]) -> dict[str, float]:
    """Run one cuperftest case and return each event's fastest run in ms."""
    out = subprocess.run(
        [binary] + args, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    # cuperftest prefixes the CSV with "# key = value" option lines.
    body = "\n".join(ln for ln in out.splitlines() if not ln.startswith("#"))
    rows = csv.DictReader(io.StringIO(body))
    times = {r["event"]: float(r[METRIC_COLUMN]) for r in rows}
    # Only the stages the two libraries share: cuperftest also times the
    # host-device transfers, and nothing reads them, so a tag that omits them
    # still plots.
    if not times.keys() >= set(CPU_STAGES):
        raise RuntimeError(f"missing events in:\n{out}")
    return times


def gpu_total(times: dict[str, float]) -> float:
    """One case's time: the stages the CPU library also has, at their fastest run.

    The same stages and the same reduction the CPU half applies to ``perftest``,
    so the two halves publish one estimator. The transfers stay out of it: they
    stage the harness's own test data, so no library change can move them, and
    on a small case they are 70-82% of the total, which left the ratio mostly
    reading a link the node shares. Every plot stacks these three stages too, so
    each label matches its bar.
    """
    return sum(times[stage] for stage in CPU_STAGES)


# nvidia-smi withholds a field a container lacks capability for as
# "[Insufficient Permissions]" or "[N/A]", so those are dropped. mig.mode tells
# a MIG slice from the whole card whose name it reports.
GPU_QUERY = (
    "name",
    "compute_cap",
    "memory.total",
    "driver_version",
    "mig.mode.current",
)


def query_gpu() -> str:
    """One line describing the card, naming only the fields the driver gives."""
    out = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY)}",
            "--format=csv,noheader",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()[0]
    fields = dict(zip(GPU_QUERY, (f.strip() for f in out.split(","))))
    known = {k: v for k, v in fields.items() if not v.startswith("[")}
    # Every other field reads as itself; a bare "Enabled" does not.
    mig = known.pop("mig.mode.current", None)
    if mig and mig.lower() == "enabled":
        mig = mig_slice() or mig
    return ", ".join(list(known.values()) + ([f"MIG {mig.lower()}"] if mig else []))


def mig_slice() -> str | None:
    """The MIG instance profile this container holds, e.g. `1g.24gb`.

    A slice is a fraction of the card, so its absolute times are not the card's.
    The profile is the one place that fraction is stated: a container running on
    a slice reports the whole card's name, and the driver withholds
    `memory.total` from it.
    """
    for line in subprocess.run(
        ["nvidia-smi", "-L"], check=True, stdout=subprocess.PIPE, text=True
    ).stdout.splitlines():
        head, _, rest = line.strip().partition(" ")
        if head == "MIG":
            return rest.split()[0]
    return None


def nvcc_version() -> str:
    """The toolkit that built the binaries, as nvcc reports itself."""
    return subprocess.run(
        ["nvcc", "--version"], check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()
