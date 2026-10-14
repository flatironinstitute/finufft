# Put cmake on a Jenkins win agent that ships none, then run the
# cross-platform quick-start test through the Visual Studio generator, which
# drives msbuild with its own environment: no vcvars needed. cmake lands in
# CI_TOOLS inside the workspace, so the second test arm skips the download.
#
# MSVC is the one piece that needs administrator rights: vs_BuildTools.exe has
# no per-user install mode. When the agent has no Visual Studio and this job
# has no admin rights, the throw names the command SCC has to run once.
#
# Env: CI_TOOLS (install root; its bin\ is on PATH through PATH+CI_TOOLS)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (-not $env:CI_TOOLS) { throw "CI_TOOLS is not set" }
$cmakeVersion = if ($env:CMAKE_VERSION) { $env:CMAKE_VERSION } else { '3.31.6' }
$tools = $env:CI_TOOLS

New-Item -ItemType Directory -Force -Path "$tools\bin", "$tools\tmp" | Out-Null

# The archive's share\ directory has to move with cmake.exe: it resolves its
# modules relative to the executable.
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    $zip = "$tools\tmp\cmake.zip"
    Invoke-WebRequest -Uri "https://github.com/Kitware/CMake/releases/download/v$cmakeVersion/cmake-$cmakeVersion-windows-x86_64.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath "$tools\tmp" -Force
    Copy-Item -Recurse -Force "$tools\tmp\cmake-$cmakeVersion-windows-x86_64\*" $tools
}

# vswhere ships with the VS installer, but its presence says nothing about a
# usable toolchain: ask it for the VC++ tools component instead.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcToolsPath = ""
if (Test-Path $vswhere) {
    $vcToolsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
}
if (-not $vcToolsPath) {
    $admin = ([Security.Principal.WindowsPrincipal] `
            [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) {
        throw @"
no Visual Studio on this agent and no administrator rights to install one.
vs_BuildTools.exe has no per-user install mode, so SCC has to run, once, as
administrator:
  winget install --id Microsoft.VisualStudio.2022.BuildTools --override ``
    "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
"@
    }
    $bootstrap = "$tools\tmp\vs_BuildTools.exe"
    Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile $bootstrap
    # 3010 is "installed, wants a reboot", which the toolchain does not need.
    & $bootstrap --quiet --wait --norestart --nocache `
        --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) {
        throw "vs_BuildTools returned $LASTEXITCODE"
    }
}

cmake -P tools/ci/install-test.cmake
if ($LASTEXITCODE -ne 0) { throw "install-test.cmake failed with $LASTEXITCODE" }
