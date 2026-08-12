#!/usr/bin/env bash
# git textconv driver: show C/C++/CUDA through clang-format so that
# formatting-only commits (see .git-blame-ignore-revs) disappear from
# `git diff`. Both sides are normalized, so the clang-format version used
# historically does not matter. Falls back to the raw file if clang-format is
# missing, so a diff never fails because of this hook.
set -u
clang-format "$1" 2>/dev/null || cat "$1"
