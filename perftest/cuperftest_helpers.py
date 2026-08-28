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


def gpu_args(param, transform: int, method: int | None = None) -> list[str]:
    """cuperftest invocation for one case of the shared CPU parameter list.

    ``method`` names one spreading method, or ``None`` to leave ``--method`` at
    the binary's own default. Two callers want different things: the PR comment
    runs one binary's default on both arms, so leaving the option out measures
    what a caller who never sets ``gpu_method`` gets. The page compares release
    tags, each built with its own cuperftest, so it names the method explicitly
    or the older tags would run a different one.
    """
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
    return (
        shared
        + [
            f"--type={transform}",
            f"--n_runs={NRUNS}",
            "--sort=1",
            "--debug=0",
        ]
        + ([] if method is None else [f"--method={method}"])
    )


def gpu_methods(transform: int) -> list[int]:
    """The spreading methods one case is charted at, so no chart asks for an empty cell.

    ``spreadinterp.cpp`` dispatches interp over methods 1 and 2 only, so a type 2
    stops at 2 and anything past it raises ``FINUFFT_ERR_METHOD_NOTVALID``. A
    type 1 spreads, so it reaches the output-driven method 3, and so does a type
    3, which spreads on its outer plan and interpolates on an inner type 2 plan
    that picks its own method. Method 0 is the library's own pick, which is what
    a caller who leaves ``gpu_method`` alone runs, and every tag the page charts
    resolves it. Method 4, block gather, is not measured at all: its kernels are
    3D spread only.
    """
    return [0, 1, 2, 3] if transform in (1, 3) else [0, 1, 2]


# The name each method goes by, from ``docs/c_gpu.rst`` and the kernel files it
# dispatches to. c_gpu.rst calls both 2 and 3 "output-block driven", so the
# short names follow ``spread_subprob`` and ``spread_output_driven`` instead,
# which tell the two apart.
GPU_METHOD_NAMES = {
    0: "automatic choice",
    1: "GM, points-driven",
    2: "SM, subproblem",
    3: "OD, output-driven",
}


def gpu_method_heading(method: int | None) -> str:
    """The heading one method's charts sit under, or "" for a backend without methods.

    The method is a heading rather than one more field in the parameter line: a
    reader scanning the page has to see which method a chart belongs to without
    reading the parameters.
    """
    if method is None:
        return ""
    return f"Method {method} ({GPU_METHOD_NAMES[method]})"


def _wrap(fields: list[str]) -> str:
    """Four fields per line, as ``Params.pretty_string`` wraps them."""
    return "\n".join(" ".join(fields[i : i + 4]) for i in range(0, len(fields), 4))


def _fields(param) -> list[str]:
    return [f"{f}:{pretty_number(getattr(param, f))}" for f in GPU_FIELDS]


def gpu_params_string(param, method: int | None = None) -> str:
    """The case in the CPU page's style, minus the CPU-only thread count.

    The method belongs in the caption because the page charts one case once per
    method, so the captions are what tells the charts apart.
    """
    return _wrap(_fields(param) + ([] if method is None else [f"method:{method}"]))


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


if __name__ == "__main__":
    from perftest_config import TRANSFORMS, Params

    flat = Params("f", 320, 320, 1, 1, 1, 1e7, 1e-4)
    cube = Params("d", 192, 192, 128, 1, 0, 8e4, 1e-7)
    # No method named leaves the binary's default in place, which is what the PR
    # comment wants: both its arms are the same binary.
    assert not [a for a in gpu_args(flat, 1) if a.startswith("--method")]
    # The page names one, or the older tags it compares against would each run
    # whichever method their own cuperftest defaulted to.
    for m in gpu_methods(1):
        assert f"--method={m}" in gpu_args(flat, 1, m), m
    # Method 0 is a method, not "unset": a falsy value must still be passed.
    assert "--method=0" in gpu_args(flat, 1, 0)
    # Interp has no method 3 kernel, so a type 2 stops at 2. Method 4 is absent
    # from every row.
    assert gpu_methods(1) == [0, 1, 2, 3], gpu_methods(1)
    assert gpu_methods(2) == [0, 1, 2], gpu_methods(2)
    assert gpu_methods(3) == [0, 1, 2, 3], gpu_methods(3)
    assert not any(4 in gpu_methods(t) for t in TRANSFORMS)
    # The captions tell one method's chart from another's.
    captions = {gpu_params_string(cube, m) for m in gpu_methods(1)}
    assert len(captions) == len(gpu_methods(1)), captions
    assert "method:" not in gpu_params_string(flat)
    # Every method the page charts has a heading, each one distinct, and it
    # names the method rather than only numbering it.
    headings = {gpu_method_heading(m) for t in TRANSFORMS for m in gpu_methods(t)}
    assert len(headings) == len(GPU_METHOD_NAMES), headings
    assert gpu_method_heading(1) == "Method 1 (GM, points-driven)"
    # A backend without methods keys its charts under a heading the page leaves
    # out, so the CPU section grows no empty section title.
    assert gpu_method_heading(None) == ""
    print("cuperftest_helpers ok: support matrix held, headings name every method")
