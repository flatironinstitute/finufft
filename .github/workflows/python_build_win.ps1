$ErrorActionPreference = "Stop"
Set-Variable -Name PYTHON -Value (Get-Command python).definition
Set-Variable -Name MSYSTEM -Value MINGW64

# Setup the make.inc file
Copy-Item -Path make.inc.windows_mingw -Destination make.inc
Add-Content -Path make.inc -Value "PYTHON=""$PYTHON"""
Add-Content -Path make.inc -Value "FFLAGS+= -fallow-argument-mismatch -march=x86-64"
Add-Content -Path make.inc -Value "CFLAGS+= -march=x86-64"
Add-Content -Path make.inc -Value "CXXFLAGS+= -march=x86-64"

# mingw gcc compiler pacth to work with python
Set-Variable cygwinccompiler_py -Value ([IO.Path]::Combine((Split-Path -Path $PYTHON), "Lib", 'distutils', 'cygwinccompiler.py'))
Remove-Item -Path $cygwinccompiler_py -Force
Copy-Item -Path .\.github\workflows\cygwinccompiler.py -Destination $cygwinccompiler_py
Set-Variable libvcruntime140_a -Value ([IO.Path]::Combine((Split-Path -Path $PYTHON), "libs", 'libvcruntime140.a'))
Copy-Item -Path .\.github\workflows\libvcruntime140.a -Destination $libvcruntime140_a

# Setup the distutils.cfg file
Set-Variable distutils_cfg -Value ([IO.Path]::Combine((Split-Path -Path $PYTHON), "Lib", 'distutils', 'distutils.cfg'))
Set-Content -Path $distutils_cfg -Value "[build]`r`ncompiler=mingw32`r`n[build_ext]`r`ncompiler=mingw32"
python -m pip install --upgrade setuptools wheel numpy pip delvewheel
if (-not $?) {throw "Failed pip install"}

# call make
Set-Variable repo_root -Value ([IO.Path]::Combine($PSScriptRoot, '..', '..'))
c:\msys64\usr\bin\env MSYSTEM=MINGW64 c:\msys64\usr\bin\bash.exe -lc "cd '$repo_root' && make python-dist"
if (-not $?) {throw "Failed make python-dist"}

# Move the required DLLs inside the wheel
$env:Path += ";C:\msys64\mingw64\bin"
Set-Variable packed_wheel -Value (get-item .\python\wheelhouse\finufft-*.whl).FullName
delvewheel repair $packed_wheel -w .\wheelhouse --add-path .\lib --no-mangle-all
if (-not $?) {throw "Failed repair wheel"}

# Cleanup
Remove-Item -Path .\python\wheelhouse -Force -Recurse
