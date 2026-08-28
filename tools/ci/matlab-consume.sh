#!/bin/bash

# Consume the CPack archive matlab-build.sh produced: unpack it, put it on the
# MATLAB path and run the interface tests against it.
#
# On Jenkins this runs in the matlab-consume image, which has MATLAB, the
# declared run-time libraries, no cmake, no CUDA toolkit and no FINUFFT file the
# archive did not put there. On the host agents it runs on the machine that built
# the archive, which is a weaker proof of the same route and the only one those
# agents can give.
#
# FINUFFT_CI_GPU=1 requires a device and the cufinufft MEX; matlab_test.m fails
# rather than skipping when either is missing.
set -eo pipefail

# The one place a license is required, and the only step that starts MATLAB. mpm
# and the images install without one, so a missing manager has to say so here
# rather than reach MATLAB as "Licensing Error -1.2", which reads like a build
# failure.
#
# Checked before -x, so the address of the license manager never reaches a
# console. Jenkins masks the credential as well; this holds outside Jenkins too.
: "${MLM_LICENSE_FILE:?set it to the <port>@<host> that \`module load matlab\` reports; on Jenkins it is the matlab-license credential}"
set -ux

rm -rf pkg && mkdir pkg
tar -xzf finufft-matlab-mex-*.tar.gz -C pkg
# The archive's top level is matlab/, holding the wrappers and the MEX.
ls -l pkg/matlab

FINUFFT_BUILD_DIR=pkg matlab -batch "run('tools/ci/matlab_test.m')"
