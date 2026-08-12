#!/usr/bin/env bash
# git textconv driver: show Python through ruff's formatter so that
# formatting-only changes disappear from `git diff` / `git log -p` and only real
# edits remain. Wire it up with tools/setup-git.sh. Falls back to the raw file
# when ruff is absent, so a diff never fails because of this hook.
set -u
ruff format --quiet - <"$1" 2>/dev/null || cat "$1"
