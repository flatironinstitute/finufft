#!/usr/bin/env bash
# Local git settings that the repo cannot ship in-tree (git config is per clone).
# Run once after cloning.
set -eu
cd "$(dirname "$0")/.."

# Skip the recorded bulk-formatting commits when blaming.
git config blame.ignoreRevsFile .git-blame-ignore-revs

# Diff Python through ruff, so formatting-only commits show up empty and code
# review sees just the real changes. Paths are opted in via .gitattributes.
git config diff.pyfmt.textconv tools/pyfmt-textconv.sh
git config diff.pyfmt.cachetextconv true

git config diff.cfmt.textconv tools/cfmt-textconv.sh
git config diff.cfmt.cachetextconv true

echo "configured: blame.ignoreRevsFile, diff.pyfmt, diff.cfmt"
