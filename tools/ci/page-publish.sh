#!/bin/bash

# Publish the page as one parentless commit on the perftest-results branch.
# A branch, unlike the PR plots: readthedocs fetches the page by branch name,
# and raw.githubusercontent resolves a custom ref only by commit SHA.
set -euxo pipefail

# The figure filenames are digests of the case, so a case keeps its filename
# from run to run. The published URLs live only as long as this commit: the
# parentless push replaces the whole tree, so a figure this run did not write
# stops resolving. Carrying master's history here only ever made it bigger.
gh auth setup-git
rm -rf ../page && mkdir -p ../page/docs/pics && cd ../page
git init -q -b perftest-results .
cp "$WORKSPACE"/docs/performance_change_summary.rst docs/
cp "$WORKSPACE"/docs/pics/perftestci_* docs/pics/
# The docs-rebuild workflow turns this push into a readthedocs build of
# "latest". A push event runs workflows from the pushed ref, so the file has
# to ride the orphan commit.
mkdir -p .github/workflows
cp "$WORKSPACE"/.github/workflows/docs-rebuild.yml .github/workflows/
git config user.name flatiron-jenkins
git config user.email flatiron-jenkins@flatironinstitute.org
git add docs .github
git commit -qm "Generated new perftest report page."
git push -f https://github.com/flatironinstitute/finufft \
	HEAD:refs/heads/perftest-results
