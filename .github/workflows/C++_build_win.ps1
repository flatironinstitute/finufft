$ErrorActionPreference = "Stop"
Set-Variable -Name MSYSTEM -Value MINGW64

# Setup the make.inc file
Copy-Item -Path make.inc.windows_msys -Destination make.inc

# call make
Set-Variable repo_root -Value ([IO.Path]::Combine($PSScriptRoot, '..', '..'))
c:\msys64\usr\bin\env MSYSTEM=MINGW64 c:\msys64\usr\bin\bash.exe -lc "cd '$repo_root' && make spreadtestall && make lib && make test"
if (-not $?) {throw "Failed make"}
