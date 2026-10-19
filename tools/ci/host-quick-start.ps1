# Run the cross-platform quick-start test on the Windows agent's own
# toolchain: MSVC from the installed Visual Studio, cmake from the bundled
# CMake component. Run from the checkout root, once per test arm.
$ErrorActionPreference = 'Stop'

# vswhere ships with the VS installer, but its presence says nothing about a
# usable toolchain: ask it for the VC++ tools component instead.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }
$vsRoot = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
$vcvars = "$vsRoot\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars - is the VC++ toolset installed?"
}

# vcvars64.bat is a batch script: load its environment by importing the set
# output of a cmd that runs it, so cl.exe reaches cmake below.
cmd /c "call `"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed with $LASTEXITCODE" }

# The bundled Ninja compiles on all cores. The old generator compiled one file
# at a time.
$ninja = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path $ninja)) {
    throw "Visual Studio Ninja not found: $ninja"
}
$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_BUILD_PARALLEL_LEVEL = $env:NUMBER_OF_PROCESSORS

# Use the CMake that Visual Studio bundles, never whatever cmake is first on PATH.
$cmake = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if (-not (Test-Path $cmake)) {
    throw "Visual Studio CMake component not installed: $cmake"
}
# Prepend the bundled bin and Ninja dirs so install-test.cmake's child cmake
# calls use the same tools.
$env:Path = "$(Split-Path -Parent $cmake);$(Split-Path -Parent $ninja);$env:Path"
Write-Host "cmake: $cmake"
Write-Host "cmake: $(& $cmake --version | Select-Object -First 1)"
Write-Host "ninja: $ninja"
Write-Host "ninja: $(& $ninja --version)"

if ($env:CC -eq 'clang-cl') {
    # This arm wants clang-cl: pin it to the LLVM that Visual Studio bundles,
    # so no other clang-cl on PATH can take the build.
    $env:CC = "$vsRoot\VC\Tools\Llvm\x64\bin\clang-cl.exe"
    $env:CXX = $env:CC
    if (-not (Test-Path $env:CC)) { throw "clang-cl.exe not found at $env:CC" }
}

& $cmake -P tools/ci/install-test.cmake
if ($LASTEXITCODE -ne 0) { throw "install-test.cmake failed with $LASTEXITCODE" }
