#!/bin/bash

# One worktree per release tag, each carrying this tree's harness.
# checkout scm gives the branch, not the tags the page plots against, and
# git worktree add needs the tag's tree here.
set -euxo pipefail

# Unshallow only when shallow, so a real failure still fails.
if [ -f "$(git rev-parse --git-dir)/shallow" ]; then
	git fetch --tags --force --unshallow
else
	git fetch --tags --force
fi
rm -rf outputs && mkdir outputs
# Every tag is timed through the current harness, so only library code differs
# between the points on a plot.
for tag in $VERSIONS; do
	rm -rf ../"$tag" && git worktree add -f -d ../"$tag" tags/"$tag"
	rm -rf ../"$tag"/perftest && cp -r perftest ../"$tag"/perftest
done
