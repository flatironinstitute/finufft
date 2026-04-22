#!/usr/bin/env python3
"""Helpers for mapping staged git hunks to line ranges in the working tree."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple


Range = Tuple[int, int]
HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def git_root() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()


def repo_relative_paths(paths: Sequence[str]) -> List[str]:
    root = git_root()
    relpaths = []
    for path in paths:
        abspath = os.path.abspath(path)
        relpath = os.path.relpath(abspath, root)
        relpaths.append(relpath.replace(os.sep, "/"))
    return relpaths


def merge_ranges(ranges: Iterable[Range]) -> List[Range]:
    merged: List[Range] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def get_staged_line_ranges(paths: Sequence[str]) -> Dict[str, List[Range]]:
    relpaths = repo_relative_paths(paths)
    if not relpaths:
        return {}

    cmd = ["git", "diff", "--cached", "--no-ext-diff", "--unified=0", "--"]
    cmd.extend(relpaths)
    diff = subprocess.check_output(cmd, text=True)

    ranges: Dict[str, List[Range]] = {}
    current_path = None

    for line in diff.splitlines():
        if line.startswith("+++ "):
            if line == "+++ /dev/null":
                current_path = None
            else:
                current_path = line[6:]
                ranges.setdefault(current_path, [])
            continue

        match = HUNK_RE.match(line)
        if not match or current_path is None:
            continue

        new_start = int(match.group("new_start"))
        new_count = int(match.group("new_count") or "1")
        if new_count == 0:
            continue

        ranges[current_path].append((new_start, new_start + new_count - 1))

    return {path: merge_ranges(path_ranges) for path, path_ranges in ranges.items()}


def overlaps(ranges: Sequence[Range], start: int, end: int) -> bool:
    for range_start, range_end in ranges:
        if start <= range_end and range_start <= end:
            return True
    return False
