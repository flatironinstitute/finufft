g++ -c -fopenmp ../cnufft/besseli.cpp
g++ -c -fopenmp ../cnufft/cnufftspread.cpp
g++ -c -fopenmp ../cnufft/cnufftspread_c.cpp
gfortran -c nufft2dnew.f 
gfortran -c dfft2d.f 
gfortran -c tempspread.f 
gfortran -c nufft2d_demof90.f 
gfortran -c prini.f 
gfortran -c nufft2df90.f 
gfortran -c dirft2d.f 
gfortran -c next235.f 
gfortran -c dfftpack.f 
gfortran -o z -fopenmp -lfftw3 -lfftw3_threads *.o -lstdc++
