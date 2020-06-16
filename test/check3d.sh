#!/bin/bash
# Standard checker for all 3d routines. Sed removes the timing lines (w/ "NU")
./finufft3d_test 5 10 20 1e3 $FINUFFT_REQ_TOL | sed '/NU/d'
# Checker for 3dmany routines for each type
./finufft3dmany_test 2 10 50 20 1e3 $FINUFFT_REQ_TOL | sed '/NU/d;/T_/d' 
