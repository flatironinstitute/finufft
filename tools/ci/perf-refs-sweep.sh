#!/bin/bash

# Sweep the plots of closed PRs, which nothing else removes.
# Only pr-<n> is swept, so refs/perftest/images survives.
set -euxo pipefail

R=flatironinstitute/finufft
gh api "repos/$R/git/matching-refs/perftest/" --jq '.[].ref' | while read -r ref; do
	case "$ref" in refs/perftest/pr-*) ;; *) continue ;; esac
	n=${ref#refs/perftest/pr-}
	# Delete only on an affirmative non-open read: a failed gh call would
	# otherwise empty the state test and delete an open PR's plots.
	state=$(gh pr view "$n" --repo "$R" --json state --jq .state) || continue
	[ "$state" = OPEN ] || gh api -X DELETE "repos/$R/git/$ref" >/dev/null
done
