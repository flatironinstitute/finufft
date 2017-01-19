#!/bin/bash
# make and run some basic complex arithmetic and RAM access speed tests.
# Barnett 1/18/17

# make
gfortran complexmulttiming.f -o complexmulttimingf -O3
g++ complexmulttiming.cpp ../../src/utils.o -o complexmulttiming -O3

# run
echo C/C++:
./complexmulttiming
echo Fortran:
./complexmulttimingf
