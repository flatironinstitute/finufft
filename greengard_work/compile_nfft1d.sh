g++ -c -fopenmp ../contrib/besseli.cpp
g++ -c -fopenmp ../src/cnufftspread.cpp
g++ -c -fopenmp ../src/cnufftspread_c.cpp
gfortran -c nufft1d_demof90.f 
gfortran -c nufft1dnew.f 
gfortran -c tempspread.f 
gfortran -c prini.f 
gfortran -c nufft1df90.f 
gfortran -c dirft1d.f 
gfortran -c next235.f 
gfortran -c dfftpack.f 
gfortran -o nfft1d_demo -fopenmp -lfftw3 -lfftw3_threads \
besseli.o cnufftspread.o cnufftspread_c.o \
nufft1d_demof90.o nufft1dnew.o tempspread.o \
prini.o nufft1df90.o dirft1d.o \
next235.o dfftpack.o -lstdc++
