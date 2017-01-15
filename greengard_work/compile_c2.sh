g++ -c -fopenmp ../contrib/besseli.cpp
g++ -c -fopenmp ../src/cnufftspread.cpp
g++ -c -fopenmp ../src/cnufftspread_c.cpp
gfortran -c tempspread.f testkernel2.f 
gfortran -o z -fopenmp -lfftw3 -lfftw3_threads \
besseli.o cnufftspread.o cnufftspread_c.o \
tempspread.o testkernel2.o -lstdc++
