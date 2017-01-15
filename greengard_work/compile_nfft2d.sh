g++ -c -fopenmp ../contrib/besseli.cpp
g++ -c -fopenmp ../src/cnufftspread.cpp
g++ -c -fopenmp ../src/cnufftspread_c.cpp
gfortran -c nufft2d_demof90.f 
gfortran -c nufft2dnew.f 
gfortran -c tempspread.f 
gfortran -c dfft2d.f 
gfortran -c prini.f 
gfortran -c nufft2df90.f 
gfortran -c dirft2d.f 
gfortran -c next235.f 
gfortran -c dfftpack.f 
gfortran -o nfft2d_demo -fopenmp -lfftw3 -lfftw3_threads \
besseli.o cnufftspread.o cnufftspread_c.o \
nufft2d_demof90.o nufft2dnew.o tempspread.o \
dfft2d.o prini.o nufft2df90.o dirft2d.o \
next235.o dfftpack.o -lstdc++
