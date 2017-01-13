g++ -c -fopenmp ../cnufft/besseli.cpp
g++ -c -fopenmp ../cnufft/cnufftspread.cpp
g++ -c -fopenmp ../cnufft/cnufftspread_c.cpp
gfortran -c nufft1dnew.f 
gfortran -c nufft1d_demof90.f 
gfortran -c prini.f 
gfortran -c nufft1df90.f 
gfortran -c dirft1d.f 
gfortran -c next235.f 
gfortran -c dfftpack.f 
gfortran -o z -fopenmp -lfftw3 -lfftw3_threads *.o -lstdc++
