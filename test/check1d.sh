#!/bin/bash
# Standard checker for all 1d routines. Sed removes the timing lines (w/ "NU")
./finufft1d_test 1e3 1e3 $FINUFFT_REQ_TOL 0 | sed '/NU/d'
# Checker for 1dmany routines for each type
./finufft1dmany_test 10 1e2 1e3 $FINUFFT_REQ_TOL 0 2 | sed '/NU/d;/T_/d' 
