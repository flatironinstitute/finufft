# Build the MATLAB MEX interface, then run its tests, on the Jenkins win agent.
#
# CPU only: the Windows agent has no card, and the GPU MEX has nothing to run
# against, so matlab_test.m is left with FINUFFT_CI_GPU unset and asserts the CPU
# MEX alone.
#
# Build only, as on Linux. matlab-consume.ps1 unpacks the archive and runs it.
#
# agent-provision.ps1 has already put cmake, ninja and MATLAB on PATH. This step
# adds the compiler, which lives in the environment vcvars sets rather than on a
# path, so it has to be entered here and cannot be set by the Jenkinsfile.
$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'vswhere found no Visual Studio with the C++ tools' }
Import-Module "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation `
    -DevCmdArguments '-arch=x64 -host_arch=x64'

# The MEX links against MATLAB's own release runtime, so the static one here too,
# and Embedded debug info because Ninja shares one .pdb across the objects.
cmake --preset matlab -B build -DFINUFFT_USE_CUDA=OFF `
    '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>' `
    -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
if ($LASTEXITCODE) { throw "cmake configure failed with $LASTEXITCODE" }
cmake --build build --target finufft_mex --parallel
if ($LASTEXITCODE) { throw "cmake build failed with $LASTEXITCODE" }

# CPack, not `cmake --install`: the archive is the artifact the MATLAB users get.
# matlab-consume.ps1 runs the tests against what comes out of it.
cpack --config build/CPackConfig.cmake -B $PWD
if ($LASTEXITCODE) { throw "cpack failed with $LASTEXITCODE" }
Get-ChildItem finufft-matlab-mex-*
