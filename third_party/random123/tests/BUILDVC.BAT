:: Call this with either x86 (for 32bit compile on a 32bit machine),
:: amd64 (if you want to do a 64bit compile on a 64bit machine),
:: or x86_amd64 (if you want to compile for amd64 on an x86)
:: With no argument, will default to amd64 on Win64 and x86 otherwise.
:: Call this with the argument "run" if you want to only run the 
:: previously compiled executables.
@echo off
setlocal
if not "%1"=="run" goto :Default
set CC=echo
set CFLAGS=
goto :Loop

:Default
if "%~1"=="" goto :Guess
set NEWBUILDVC=%1
goto :Next

:Guess
set RegQry=HKLM\Hardware\Description\System\CentralProcessor\0
REG.exe Query %RegQry% > hwdesc.o
FIND /i "x86" < hwdesc.o > hwcheck.o
if %errorlevel% == 0 (
set NEWBUILDVC=x86
) else (
set NEWBUILDVC=amd64
)

:Next
if "%BUILDVC%"=="%NEWBUILDVC%" goto :Continue
if NOT DEFINED VCBAT set VCBAT="c:\Program Files (x86)\Microsoft Visual Studio 10.0\vc\vcvarsall.bat"
call %VCBAT% %NEWBUILDVC%
if errorlevel 1 exit /b 1
set BUILDVC=%NEWBUILDVC%

:Continue
:: /Zi for debug.  /favor:INTEL64 is ignored for 32bit compiles.
if NOT DEFINED CFLAGS set CFLAGS=/DR123_NO_SINCOS=1 /I..\include /W3 /Ox /EHs /nologo /favor:INTEL64
set CC=cl
echo Using %VCBAT%
echo Building for %BUILDVC% with %CC% %CFLAGS%

:Loop
set BUILDFILES= ( kat_c.c kat_cpp.cpp pi_aes.cpp pi_capi.c pi_cppapi.cpp pi_microurng.cpp simple.c simplepp.cpp time_serial.c time_boxmuller.cpp timers.cpp ut_Engine.cpp ut_M128.cpp ut_ReinterpretCtr.cpp ut_aes.cpp ut_ars.c ut_carray.cpp ut_features.cpp ut_uniform.cpp )
FOR %%A IN %BUILDFILES% DO (
	%CC% %CFLAGS% %%A
	if errorlevel 1 exit /b 1
	%%~nA 
	if errorlevel 1 exit /b 1 )
endlocal
