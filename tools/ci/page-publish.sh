#!/bin/bash

# Publish the page as one parentless commit on the perftest-results branch.
# A branch, unlike the PR plots: readthedocs fetches the page by branch name,
# and raw.githubusercontent resolves a custom ref only by commit SHA.
set -euxo pipefail

# The figure filenames are digests of the case, so the URLs already published
# keep resolving; carrying master's history here only ever made it bigger.
gh auth setup-git
rm -rf ../page && mkdir -p ../page/docs/pics && cd ../page
git init -q -b perftest-results .
cp "$WORKSPACE"/docs/performance_change_summary.rst docs/
cp "$WORKSPACE"/docs/pics/perftestci_* docs/pics/
git config user.name flatiron-jenkins
git config user.email flatiron-jenkins@flatironinstitute.org
git add docs
git commit -qm "Generated new perftest report page."
git push -f https://github.com/flatironinstitute/finufft \
	HEAD:refs/heads/perftest-results
