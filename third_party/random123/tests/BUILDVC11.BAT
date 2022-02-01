rem /DR123_USE_CXX11=1 will not work because VS2012 is not fully C++11 compliant yet.
setlocal
set CFLAGS=/I..\include /W3 /Ox /EHs /nologo /favor:INTEL64
set VCBAT="c:\Program Files (x86)\Microsoft Visual Studio 11.0\vc\vcvarsall.bat"
call BUILDVC.bat %1
endlocal
