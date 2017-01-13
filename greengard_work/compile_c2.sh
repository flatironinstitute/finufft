g++ -c -fopenmp ../cnufft/besseli.cpp
g++ -c -fopenmp ../cnufft/cnufftspread.cpp
g++ -c -fopenmp ../cnufft/cnufftspread_c.cpp
gfortran -c tempspread.f testkernel2.f 
gfortran -o z -fopenmp -lfftw3 -lfftw3_threads *.o -lstdc++
