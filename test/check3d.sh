#!/bin/bash
# Standard checker for all 3d routines. Sed removes the timing lines (w/ "NU")
./finufft3d_test 5 10 20 1e3 $FINUFFT_REQ_TOL 0 | sed '/NU/d'
#Checker for 3dmany routines for each type
./finufft3dmany_test 10 1e2 1e2 1e1 1e3 $FINUFFT_REQ_TOL 0 2 | sed '/NU/d;/T_/d' 
