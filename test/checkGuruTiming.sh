#!/bin/bash

#------------------------------------------------------------#1D

: ' #uncomment to start comment over 1D section 

#********************T1

######1
echo "./finufftGuru_test 1 1 1 1e2 1 1 1e-6 1"
./finufftGuru_test 1 1 1 1e2 1 1 1e7 1e-6 1

#######10
echo "./finufftGuru_test 10 1 1 1e2 1 1 1e-6 1"
./finufftGuru_test 10 1 1 1e2 1 1 1e7 1e-6 1

#######100
echo "./finufftGuru_test 100 1 1 1e2 1 1 1e-6 1" 
./finufftGuru_test 100 1 1 1e2 1 1 1e7 1e-6 1

#********************T2

#######1
echo "./finufftGuru_test 1 2 1 1e2 1 1 1e-6 1"
./finufftGuru_test 1 2 1 1e2 1 1 1e7 1e-6 1

#######10
echo "./finufftGuru_test 10 2 1 1e2 1 1 1e-6 1"
./finufftGuru_test 10 2 1 1e2 1 1 1e7 1e-6 1

#######100
echo "./finufftGuru_test 100 2 1 1e2 1 1 1e-6 1" 
./finufftGuru_test 100 2 1 1e2 1 1 1e7 1e-6 1

#********************T3

#######1
echo "./finufftGuru_test 1 3 1 1e2 1 1 1e-6 1" 
./finufftGuru_test 1 3 1 1e2 1 1 1e7 1e-6 1

#######10
echo "./finufftGuru_test 10 3 1 1e2 1 1 1e-6 1" 
./finufftGuru_test 10 3 1 1e2 1 1 1e7 1e-6 1

#######100 
echo "./finufftGuru_test 100 3 1 1e2 1 1 1e-6 1"
./finufftGuru_test 100 3 1 1e2 1 1 1e7 1e-6 1

' # end of commented out 1D

#------------------------------------------------------------#2D

: ' #uncomment to start comment over 2D section 

#********************T1

#######1
echo "./finufftGuru_test 1 1 2 1e2 1e2 1 1e7 1e-6 1"
./finufftGuru_test 1 1 2 1e2 1e2 1 1e7 1e-6 1

#######10
echo "./finufftGuru_test 10 1 2 1e2 1e2 1 1e7 1e-6 1" 
./finufftGuru_test 10 1 2 1e2 1e2 1 1e7 1e-6 1

#######100
echo "./finufftGuru_test 100 1 2 1e2 1e2 1 1e7 1e-6 1" 
./finufftGuru_test 100 1 2 1e2 1e2 1 1e7 1e-6 1


#********************T2

#######1
echo "./finufftGuru_test 1 2 2 1e2 1e2 1 1e7 1e-6 1" 
./finufftGuru_test 1 2 2 1e2 1e2 1 1e7 1e-6 1

#######10
echo "./finufftGuru_test 10 2 2 1e2 1e2 1 1e7 1e-6 1" 
./finufftGuru_test 10 2 2 1e2 1e2 1 1e7 1e-6 1

#######100
echo "./finufftGuru_test 100 2 2 1e2 1e2 1 1e7 1e-6 1"
./finufftGuru_test 100 2 2 1e2 1e2 1 1e7 1e-6 1

#********************T3

#######1
echo "./finufftGuru_test 1 3 2 1e2 1e2 1 1e7 1e-6 1" 
./finufftGuru_test 1 3 2 1e2 1e2 1 1e7 1e-6 1
 
#######10
echo "./finufftGuru_test 10 3 2 1e2 1e2 1 1e7 1e-6 1" 
./finufftGuru_test 10 3 2 1e2 1e2 1 1e7 1e-6 1

#######100
echo "./finufftGuru_test 100 3 2 1e2 1e2 1 1e7 1e-6 1"
./finufftGuru_test 100 3 2 1e2 1e2 1 1e7 1e-6 1

 ' # end of commented out 1D

#------------------------------------------------------------#3D

#: ' #uncomment to start comment over 3D section 

#********************T1

#######1
echo "./finufftGuru_test 1 1 3 1e2 1e2 1e2 1e7 1e-6 1"
./finufftGuru_test 1 1 3 1e2 1e2 1e2 1e7 1e-6 1

#######10
echo "./finufftGuru_test 10 1 3 1e2 1e2 1e2 1e7 1e-6 1" 
./finufftGuru_test 10 1 3 1e2 1e2 1e2 1e7 1e-6 1

#######100
echo "./finufftGuru_test 100 1 3 1e2 1e2 1e2 1e7 1e-6 1" 
./finufftGuru_test 100 1 3 1e2 1e2 1e2 1e7 1e-6 1


#********************T2

#######1
echo "./finufftGuru_test 1 2 3 1e2 1e2 1e2 1e7 1e-6 1"
./finufftGuru_test 1 2 3 1e2 1e2 1e2 1e7 1e-6 1

#######10
echo "./finufftGuru_test 10 2 3 1e2 1e2 1e2 1e7 1e-6 1"
./finufftGuru_test 10 2 3 1e2 1e2 1e2 1e7 1e-6 1

#######100
echo "./finufftGuru_test 100 2 3 1e2 1e2 1e2 1e7 1e-6 1" 
./finufftGuru_test 100 2 3 1e2 1e2 1e2 1e7 1e-6 1

#********************T3

#######1
echo "./finufftGuru_test 1 3 3  1e2 1e2 1e2 1e7 1e-6 1"
./finufftGuru_test 1 3 3  1e2 1e2 1e2 1e7 1e-6 1

#######10
echo "./finufftGuru_test 10 3 3 1e2 1e2 1e2 1e7 1e-6 1"
./finufftGuru_test 10 3 3 1e2 1e2 1e2 1e7 1e-6 1

#######100
echo "./finufftGuru_test 100 3 3 1e2 1e2 1e2 1e7 1e-6 1" 
./finufftGuru_test 50 3 3 1e2 1e2 1e2 1e7 1e-6 1

#'
