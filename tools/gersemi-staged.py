#!/usr/bin/env python3
"""Format staged CMake hunks only by forwarding ranges to gersemi."""

from __future__ import annotations

import os
import subprocess
import sys

from staged_ranges import get_staged_line_ranges
from staged_ranges import repo_relative_paths


def main() -> int:
    paths = sys.argv[1:]
    ranges_by_path = get_staged_line_ranges(paths)
    modified = []

    for path, relpath in zip(paths, repo_relative_paths(paths)):
        ranges = ranges_by_path.get(relpath, [])
        if not ranges:
            continue

        before = open(path, "rb").read()
        range_arg = ",".join(f"{start}-{end}" for start, end in ranges)
        result = subprocess.run(
            ["gersemi", "--in-place", "--line-ranges", range_arg, path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            if output:
                print(output, file=sys.stderr)
            return result.returncode

        after = open(path, "rb").read()
        if after != before:
            modified.append(path)

    if modified:
        for path in modified:
            print(f"reformatted staged CMake hunks in {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
