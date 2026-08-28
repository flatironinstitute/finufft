# Put cmake, ninja, MATLAB and the MSVC build tools on the Jenkins win agent,
# which was probed on 2026-08-28 and carries none of them: no cl.exe, no
# vswhere.exe, no cmake, no ninja, no MATLAB.
#
# Everything except MSVC lands in CI_TOOLS, outside the workspace, so the ~10 GB
# MATLAB install survives the next build, and each step is skipped when its
# target is already there.
#
# MSVC is the one piece that needs administrator rights: vs_BuildTools.exe has
# no per-user install mode. The script reports whether the agent holds them
# before it tries, so a failure names the cause instead of an installer exit
# code.
#
# Environment: CI_TOOLS, MATLAB_RELEASE
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # the progress bar costs minutes over a slow link

foreach ($v in 'CI_TOOLS', 'MATLAB_RELEASE') {
    if (-not (Get-Item "env:$v" -ErrorAction SilentlyContinue)) { throw "$v is not set" }
}
$tools = $env:CI_TOOLS
$cmakeVersion = if ($env:CMAKE_VERSION) { $env:CMAKE_VERSION } else { '3.31.6' }
$ninjaVersion = if ($env:NINJA_VERSION) { $env:NINJA_VERSION } else { '1.12.1' }
$products = if ($env:MATLAB_PRODUCTS) { $env:MATLAB_PRODUCTS } else { 'MATLAB Parallel_Computing_Toolbox' }

New-Item -ItemType Directory -Force -Path "$tools\bin", "$tools\tmp" | Out-Null

# What this agent is, before anything can fail. The MSVC question is answered
# here rather than at the bottom, so one build reports it even when an earlier
# step dies: vs_BuildTools.exe has no per-user install mode, and whether SCC has
# to run it by hand turns entirely on this line.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$admin = ([Security.Principal.WindowsPrincipal] `
        [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Write-Host "agent: $env:USERNAME, administrator: $admin, vswhere: $(Test-Path $vswhere), tools: $tools"

# cmake and ninja both end up under $tools\bin, so one PATH entry covers the
# toolchain. cmake resolves its modules relative to the executable, so the
# archive's share\ directory has to move with it.
if (-not (Test-Path "$tools\bin\cmake.exe")) {
    $zip = "$tools\tmp\cmake.zip"
    Invoke-WebRequest -Uri "https://github.com/Kitware/CMake/releases/download/v$cmakeVersion/cmake-$cmakeVersion-windows-x86_64.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath "$tools\tmp" -Force
    Copy-Item -Recurse -Force "$tools\tmp\cmake-$cmakeVersion-windows-x86_64\*" $tools
}

if (-not (Test-Path "$tools\bin\ninja.exe")) {
    $zip = "$tools\tmp\ninja.zip"
    Invoke-WebRequest -Uri "https://github.com/ninja-build/ninja/releases/download/v$ninjaVersion/ninja-win.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath "$tools\bin" -Force
}

# mpm resumes nothing: a half-finished install would be taken for a good one, so
# it builds under .partial and is renamed only once mpm has returned 0.
$matlabRoot = "$tools\matlab\$env:MATLAB_RELEASE"
if (-not (Test-Path "$matlabRoot\bin\matlab.exe")) {
    Remove-Item -Recurse -Force "$matlabRoot.partial" -ErrorAction SilentlyContinue
    $mpm = "$tools\tmp\mpm.exe"
    Invoke-WebRequest -Uri 'https://www.mathworks.com/mpm/win64/mpm' -OutFile $mpm
    & $mpm install --release=$env:MATLAB_RELEASE --destination="$matlabRoot.partial" --products $products.Split(' ')
    if ($LASTEXITCODE -ne 0) { throw "mpm install failed with $LASTEXITCODE" }
    Move-Item "$matlabRoot.partial" $matlabRoot
}

# Assert rather than trust: a truncated install has to fail here, not later in a
# stage where it reads as a code failure. No `matlab -batch`, which would need a
# license this step is not given.
foreach ($f in "$matlabRoot\bin\matlab.exe", "$matlabRoot\bin\mex.bat",
    "$matlabRoot\extern\include\mex.h",
    "$matlabRoot\toolbox\parallel\gpu\extern\include\gpu\mxGPUArray.h") {
    if (-not (Test-Path $f)) { throw "$f missing after provisioning" }
}

if (-not (Test-Path $vswhere)) {
    if (-not $admin) {
        throw @"
no Visual Studio on this agent and no administrator rights to install one.
vs_BuildTools.exe has no per-user install mode, so SCC has to run, once, as
administrator:
  winget install --id Microsoft.VisualStudio.2022.BuildTools --override `
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

& "$tools\bin\cmake.exe" --version | Select-Object -First 1
& "$tools\bin\ninja.exe" --version
Write-Host "MATLAB $env:MATLAB_RELEASE at $matlabRoot"
& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
