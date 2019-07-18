#!/bin/bash

# [ntransf [type [ndim [N1 N2 N3 [Nsrc [tol [debug [do_spread [upsamplefac]]]]]]]]]

#in order to not suppress output of finufft?d?_old comparison, must comment out top lines doing so in runOldFinufft.cpp


#./finufftGuru_test 1 1 3 1e2 1e2 1e2 750 1e-6 1

#./finufftGuru_test 25 1 2 1e3 1e3 0 1e4 1e-6 1

#./finufftGuru_test 1 1 3 1e3 1e2 1e1 10 1e-6 1 #low density

#./finufftGuru_test 1 1 3 1 1 100 20000 1e-6 1 #very bad. believe deconvolve/shuffle is bottleneck (10x slower)

#./finufftGuru_test 1 1 1 1 0 0 20000 1e-6 1 #same as above


#./finufftGuru_test 1 2 3 1e2 1e2 1e4 750 1e-6 1

#./finufftGuru_test 1 2 3 1e3 1e3 1e6 15 1e-6 1

#./finufftGuru_test 1 2 3 1e3 1e3 1e1 10 1e-6 1 #low density

#./finufftGuru_test 1 2 3 1e1 1e1 1e2 20000 1e-6 1



#./finufftGuru_test 1 3 3 1e2 1e2 1e2 750 1e-6 1

#./finufftGuru_test 15 3 2 1e3 1e3 0 1e5 1e-6 1

#./finufftGuru_test 10 3 2 1e3 1e1 0 1e3 1e-6 1 #low density

#./finufftGuru_test 200 3 2 1e1 1e1 1e2 0 1e-6 1


