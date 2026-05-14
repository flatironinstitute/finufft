#!/usr/bin/env python3
"""Generate a "Supported CPU Platforms" table from the CI workflows.

Reads `.github/workflows/cmake_ci.yml` and `.github/workflows/powerpc.yml`,
extracts their matrix rows, and emits a markdown or reStructuredText table
whose status cells are live shields.io badges linked to the workflow runs.

The committed CI matrix is the only source of truth; this script never
writes back. See issue #508 for context.
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

REPO = "flatironinstitute/finufft"
BRANCH = "master"  # overridden by --branch
BADGE_STYLE = "flat-square"
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


@dataclass(frozen=True)
class Row:
    os: str
    compiler: str
    workflow: str  # workflow file name, e.g. "cmake_ci.yml"
    job_name: str  # job name as rendered by GitHub Actions


def _badge_url(workflow: str, job_name: str, label: str = "") -> str:
    # The /github/actions/workflow/status endpoint reports the WORKFLOW-level
    # status and silently ignores a ?job= filter, so every per-job badge would
    # collapse to the same colour. /github/check-runs supports nameFilter and
    # returns the per-check-run conclusion (each matrix job is one check run).
    ref = urllib.parse.quote(BRANCH, safe="")
    q = urllib.parse.urlencode(
        {"nameFilter": job_name, "label": label, "style": BADGE_STYLE},
        quote_via=urllib.parse.quote,
    )
    del workflow  # retained in the call signature for link generation
    return f"https://img.shields.io/github/check-runs/{REPO}/{ref}?{q}"


def _runs_url(workflow: str) -> str:
    return f"https://github.com/{REPO}/actions/workflows/{workflow}?query=branch%3A{BRANCH}"


def parse_cmake_ci(path: Path) -> list[Row]:
    data = yaml.safe_load(path.read_text())
    include = data["jobs"]["cmake-ci"]["strategy"]["matrix"]["include"]
    rows = []
    for entry in include:
        os_ = entry["os"]
        tc = entry["toolchain"]
        # GitHub Actions joins matrix.include values in declaration order
        # separated by ", " to form the job name suffix; the badge job filter
        # must match that exactly.
        job_name = f"cmake-ci ({os_}, {tc})"
        # setup-cpp's `llvm` keyword installs LLVM clang; show it as `clang`
        # in the table so macOS rows don't list both `llvm` and `clang` for
        # the same compiler family.
        display = "clang" if tc == "llvm" else tc
        rows.append(
            Row(os=os_, compiler=display, workflow="cmake_ci.yml", job_name=job_name)
        )
    return rows


def parse_cpp_make(path: Path) -> list[Row]:
    """Rows from C++.yml — the legacy GNU-make build path.

    Each job is a single non-matrix runs-on. The job name as rendered by GH
    Actions is just the YAML job key.
    """
    data = yaml.safe_load(path.read_text())
    jobs = data["jobs"]
    # Hand-pick the compiler label since each job hard-codes its own toolchain.
    # (os override, compiler label) — os override is used when the job runs
    # inside a container or shell whose environment differs from runs-on.
    overrides = {
        "Linux": ("manylinux_2_28", "gcc (make)"),
        "MacOS_clang": (None, "clang (make)"),
        "MacOS_gcc": (None, "gcc-15 (make)"),
        "Windows": ("MSYS2 MINGW64", "gcc (make)"),
    }
    rows = []
    for key, spec in jobs.items():
        if key not in overrides:
            continue
        os_override, compiler = overrides[key]
        rows.append(
            Row(
                os=os_override or spec["runs-on"],
                compiler=compiler,
                workflow="C++.yml",
                job_name=key,
            )
        )
    return rows


def parse_powerpc(path: Path) -> list[Row]:
    data = yaml.safe_load(path.read_text())
    job = data["jobs"]["build"]
    matrix = job["strategy"]["matrix"]
    rows = []
    for target in matrix["target"]:
        for sysd in matrix["sys"]:
            # powerpc.yml sets a custom job `name:` template:
            #   '${{ matrix.target.arch }}, ${{ matrix.sys.compiler }} ${{ matrix.sys.version }}'
            base = f"{target['arch']}, {sysd['compiler']} {sysd['version']}"
            job_name = f"build ({base})"
            arch = target["arch"]
            os_label = "powerpc64le" if arch == "ppc64le" else "powerpc64"
            rows.append(
                Row(
                    os=os_label,
                    compiler=f"{sysd['compiler']}-{sysd['version']} (cross)",
                    workflow="powerpc.yml",
                    job_name=job_name,
                )
            )
    return rows


def collect_rows() -> list[Row]:
    rows = parse_cmake_ci(WORKFLOWS / "cmake_ci.yml")
    rows += parse_cpp_make(WORKFLOWS / "C++.yml")
    rows += parse_powerpc(WORKFLOWS / "powerpc.yml")
    rows.sort(key=lambda r: (r.os, r.compiler))
    return rows


def _group_by_os(rows: Iterable[Row]) -> "list[tuple[str, list[Row]]]":
    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r.os, []).append(r)
    return sorted(groups.items())


def render_md(rows: Iterable[Row]) -> str:
    rows = list(rows)
    groups = _group_by_os(rows)
    out = [
        "| Platform | Toolchains (per-job CI status) |",
        "|----------|--------------------------------|",
    ]
    for os_, group in groups:
        cells = []
        for r in group:
            badge = _badge_url(r.workflow, r.job_name, label=r.compiler)
            link = _runs_url(r.workflow)
            cells.append(f"[![{r.compiler}]({badge})]({link})")
        out.append(f"| {os_} | {' '.join(cells)} |")
    out.append("")
    out.append(
        "_Each badge labels its toolchain and colours green/red with that job's "
        "current status on `master`. CMake-CI rows on Linux/macOS exercise both "
        "DUCC FFT and FFTW; Windows builds DUCC FFT only. `(make)` rows exercise "
        "the legacy GNU-make build path. PowerPC rows build via cross-compile + QEMU._"
    )
    return "\n".join(out) + "\n"


def render_rst(rows: list[Row]) -> str:
    # Multi-line image directives don't fit cleanly inside list-table cells;
    # define each badge once as a substitution and reference it from the cell.
    subs = []
    refs = []
    for i, r in enumerate(rows):
        badge = _badge_url(r.workflow, r.job_name)
        link = _runs_url(r.workflow)
        name = f"badge-{i}"
        subs.append(
            f".. |{name}| image:: {badge}\n   :target: {link}\n   :alt: {r.job_name}"
        )
        refs.append(name)

    lines = [
        "Supported CPU platforms",
        "-----------------------",
        "",
    ]
    lines.extend(s + "\n" for s in subs)
    lines += [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 25 25 50",
        "",
        "   * - OS",
        "     - Compiler",
        "     - CI status",
    ]
    for r, name in zip(rows, refs):
        lines.append(f"   * - {r.os}")
        lines.append(f"     - {r.compiler}")
        lines.append(f"     - |{name}|")
    lines.append("")
    lines.append(
        "*CMake CI rows: each Linux/macOS job builds and tests both DUCC FFT and "
        "FFTW; Windows jobs build DUCC FFT only. Rows labelled* ``(make, ...)`` "
        "*exercise the legacy GNU-make build path. PowerPC rows build via "
        "cross-compile + QEMU.*"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=("md", "rst"), default="md")
    p.add_argument(
        "--branch",
        default=None,
        help="branch the badges should report status for (default: master)",
    )
    p.add_argument(
        "--repo",
        default=None,
        help="owner/repo the badges should query (default: flatironinstitute/finufft)",
    )
    p.add_argument("--output", type=Path, help="write to file instead of stdout")
    p.add_argument(
        "--check",
        action="store_true",
        help="parse the workflows and exit non-zero on failure; no output",
    )
    p.add_argument(
        "--inject",
        type=Path,
        help=(
            "replace the block between "
            "'<!-- BEGIN: generated by tools/gen_platform_table.py -->' and "
            "'<!-- END: generated by tools/gen_platform_table.py -->' in PATH "
            "(markdown only)"
        ),
    )
    args = p.parse_args()

    global BRANCH, REPO
    if args.branch:
        BRANCH = args.branch
    if args.repo:
        REPO = args.repo

    rows = collect_rows()
    if not rows:
        print("error: no matrix rows extracted", file=sys.stderr)
        return 1

    if args.check:
        return 0

    text = render_md(rows) if args.format == "md" else render_rst(rows)

    if args.inject is not None:
        begin = "<!-- BEGIN: generated by tools/gen_platform_table.py -->"
        end = "<!-- END: generated by tools/gen_platform_table.py -->"
        src = args.inject.read_text()
        try:
            head, _rest = src.split(begin, 1)
            _old, tail = _rest.split(end, 1)
        except ValueError:
            print(
                f"error: markers not found in {args.inject}; expected\n  {begin}\n  {end}",
                file=sys.stderr,
            )
            return 2
        args.inject.write_text(f"{head}{begin}\n\n{text}\n{end}{tail}")
        return 0

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
