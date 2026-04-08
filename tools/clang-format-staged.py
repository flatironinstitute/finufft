#!/usr/bin/env python3
"""Wrapper around git-clang-format that cleans up stale .lock files.

git-clang-format uses a fixed temp-index path (.git/clang-format-index)
rather than a unique temp file. When pre-commit's concurrent git
operations race with it, the .lock file persists and blocks all
subsequent commits. This wrapper cleans up stale locks before and after
each run.

See: https://github.com/llvm/llvm-project/issues/52644
"""

import os
import subprocess
import sys


def git_dir():
    return subprocess.check_output(["git", "rev-parse", "--git-dir"], text=True).strip()


def remove_lock(lockpath):
    try:
        os.remove(lockpath)
    except FileNotFoundError:
        pass


def main():
    lock = os.path.join(git_dir(), "clang-format-index.lock")

    # Remove stale lock from a previous interrupted run.
    remove_lock(lock)

    cmd = ["git-clang-format", "--binary", "clang-format", "--staged", "--"]
    cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up lock in case it leaked.
    remove_lock(lock)

    output = (result.stdout + result.stderr).strip()

    if result.returncode != 0:
        if output:
            print(output, file=sys.stderr)
        sys.exit(result.returncode)

    if "clang-format did not modify any files" in output:
        sys.exit(0)

    # Files were modified — print what changed and fail so pre-commit re-stages.
    if output:
        print(output)
    sys.exit(1)


if __name__ == "__main__":
    main()
