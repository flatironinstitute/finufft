#!/usr/bin/env python3
"""Format staged CMake hunks only by splicing full-file gersemi output.

gersemi's own --line-ranges rewrites lines outside the given ranges, and the
rewrites do not converge across runs (each re-staging shifts the ranges and
flips the same lines again), so the hook cannot trust it. The file is
formatted in full instead and only the hunks intersecting the staged lines
are spliced back in; everything outside them stays byte-identical, the same
contract the clang-format hook keeps for C++.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys

from staged_ranges import get_staged_line_ranges
from staged_ranges import repo_relative_paths


def intersects(ranges, i1: int, i2: int) -> bool:
    """Whether the diff hunk covering original lines [i1, i2) (0-based)
    touches a staged range (1-based, inclusive)."""
    if i1 < i2:  # replace/delete: covers original lines i1..i2-1
        return any(s - 1 <= i2 - 1 and e - 1 >= i1 for s, e in ranges)
    # insertion before original line i1: near a staged line counts as touching it
    return any(s - 1 <= i1 <= e for s, e in ranges)


def main() -> int:
    paths = sys.argv[1:]
    ranges_by_path = get_staged_line_ranges(paths)
    modified = []

    for path, relpath in zip(paths, repo_relative_paths(paths)):
        ranges = ranges_by_path.get(relpath, [])
        if not ranges or not os.path.exists(path):
            continue

        original = open(path).read()
        result = subprocess.run(["gersemi", path], capture_output=True, text=True)
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            if output:
                print(output, file=sys.stderr)
            return result.returncode
        formatted = result.stdout
        if formatted == original:
            continue

        # Keep the formatted text where it lands on staged lines, the original
        # everywhere else. SequenceMatcher opcodes are quoted in original-line
        # coordinates, which is the coordinate system of the staged ranges.
        orig_lines = original.splitlines(keepends=True)
        fmt_lines = formatted.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, orig_lines, fmt_lines, autojunk=False)
        spliced = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal" or not intersects(ranges, i1, i2):
                spliced.extend(orig_lines[i1:i2])
            else:
                spliced.extend(fmt_lines[j1:j2])
        spliced_text = "".join(spliced)

        if spliced_text != original:
            with open(path, "w") as f:
                f.write(spliced_text)
            modified.append(path)

    if modified:
        for path in modified:
            print(f"reformatted staged CMake hunks in {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
