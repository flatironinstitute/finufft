g++ -c -fopenmp ../contrib/besseli.cpp
g++ -c -fopenmp ../src/cnufftspread.cpp
g++ -c -fopenmp ../src/cnufftspread_c.cpp
gfortran -c nufft3d_demof90.f 
gfortran -c nufft3dnew.f 
gfortran -c tempspread.f 
gfortran -c dfft3d.f 
gfortran -c prini.f 
gfortran -c nufft3df90.f 
gfortran -c dirft3d.f 
gfortran -c next235.f 
gfortran -c dfftpack.f 
gfortran -o nfft3d_demo -fopenmp -lfftw3 -lfftw3_threads \
besseli.o cnufftspread.o cnufftspread_c.o \
nufft3d_demof90.o nufft3dnew.o tempspread.o \
dfft3d.o prini.o nufft3df90.o dirft3d.o \
next235.o dfftpack.o -lstdc++
