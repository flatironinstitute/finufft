#!/bin/bash

# Push the two plots as a parentless commit and point the comment body at them.
# No API attaches an image to a comment, so they ride on refs/perftest/pr-<n>,
# linked through raw.githubusercontent. A clone follows only heads and tags.
set -euxo pipefail

R=flatironinstitute/finufft
entry() {
	sha=$(printf '{"encoding":"base64","content":"%s"}' "$(base64 -w0 "$2")" |
		gh api -X POST "repos/$R/git/blobs" --input - --jq .sha)
	printf '{"path":"%s","mode":"100644","type":"blob","sha":"%s"}' "$1" "$sha"
}
tree=$(printf '{"tree":[%s,%s]}' "$(entry cpu.svg cpu_perf.svg)" \
	"$(entry gpu.svg gpu_perf.svg)" |
	gh api -X POST "repos/$R/git/trees" --input - --jq .sha)
head=$(printf '{"message":"perftest plots PR #%s","tree":"%s","parents":[]}' \
	"$CHANGE_ID" "$tree" |
	gh api -X POST "repos/$R/git/commits" --input - --jq .sha)
# Force-updating the ref strands the old plots for GitHub to collect, so
# storage stays at two SVGs per PR.
ref=perftest/pr-$CHANGE_ID
gh api -X PATCH "repos/$R/git/refs/$ref" -F sha="$head" -F force=true >/dev/null 2>&1 ||
	gh api -X POST "repos/$R/git/refs" -f ref="refs/$ref" -f sha="$head" >/dev/null
base=https://raw.githubusercontent.com/$R/$head
sed -i -e "s|CPU_IMAGE_URL|$base/cpu.svg|" -e "s|GPU_IMAGE_URL|$base/gpu.svg|" perf_body.md
