# Consume the CPack archive matlab-build.ps1 produced: unpack it, put it on the
# MATLAB path and run the interface tests against it.
#
# The Windows agent builds and consumes on one machine. Linux splits the two
# across images, one with the toolchain and one with MATLAB alone; Windows has
# no second agent to be the clean room, so this proves the archive route without
# proving the absence of a toolchain.
$ErrorActionPreference = 'Stop'

# The one step that starts MATLAB, so the one that needs a license. mpm installs
# without one, and a missing manager has to say so here rather than reach MATLAB
# as "Licensing Error -1.2", which reads like a build failure.
if (-not $env:MLM_LICENSE_FILE) {
    throw 'MLM_LICENSE_FILE is not set: give it the <port>@<host> that `module load matlab` reports, in the Jenkins global environment or on the agent'
}

Remove-Item -Recurse -Force pkg -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path pkg | Out-Null
$archive = (Get-ChildItem finufft-matlab-mex-*.zip).FullName
Expand-Archive -Path $archive -DestinationPath pkg -Force
# The archive's top level is matlab\, holding the wrappers and the MEX.
Get-ChildItem pkg\matlab

$env:FINUFFT_BUILD_DIR = 'pkg'
matlab -batch "run('tools/ci/matlab_test.m')"
if ($LASTEXITCODE) { throw "matlab -batch failed with $LASTEXITCODE" }
