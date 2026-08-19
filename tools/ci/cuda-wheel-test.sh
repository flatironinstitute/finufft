#!/bin/bash
# Run the installed wheel's tests against every GPU framework.
set -euxo pipefail

# activate reads variables it does not set, which set -u would call an error.
set +u && source "$HOME/venv/bin/activate" && set -u
python3 -c "from numba import cuda; cuda.cudadrv.libs.test()"
# From a copy outside the repo: pytest prepends the tests' rootdir to sys.path,
# which would shadow the installed wheel with the source tree. examples/ stays a
# sibling of tests/ (test_examples finds it relative to itself).
rm -rf "$HOME/wheeltest" && mkdir "$HOME/wheeltest"
cp -r python/cufinufft/tests python/cufinufft/examples "$HOME/wheeltest/"
cd "$HOME/wheeltest"
# Every framework runs even if an earlier one fails, so the log says which ones
# are broken rather than only the first.
rc=0
for framework in pycuda numba cupy torch; do
	python3 -m pytest --framework=$framework tests || rc=$?
done
exit $rc
