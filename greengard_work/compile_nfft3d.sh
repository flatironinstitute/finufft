g++ -c -fopenmp ../besseli.cpp
g++ -c -fopenmp ../cnufftspread.cpp
g++ -c -fopenmp ../cnufftspread_c.cpp
gfortran -c nufft3dnew.f 
gfortran -c dfft3d.f 
gfortran -c nufft3df90.f 
gfortran -c nufft3d_demof90.f 
gfortran -c tempspread.f 
gfortran -c prini.f 
gfortran -c dirft3d.f 
gfortran -c next235.f 
gfortran -c dfftpack.f 
gfortran -o z -fopenmp -lfftw3 -lfftw3_threads *.o -lstdc++
