#!/bin/bash
# Standard checker for all 2d routines. Sed removes the timing lines (w/ "NU")
./finufft2d_test 1e2 1e1 1e3 $FINUFFT_REQ_TOL | sed '/NU/d'
# Melody's addition of the "many" interface test...
./finufft2dmany_test 3 1e2 1e1 1e3 $FINUFFT_REQ_TOL | sed '/NU/d;/T_/d' 
