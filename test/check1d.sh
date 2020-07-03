#!/bin/bash
# Standard checker for all 1d routines. Sed removes the timing lines (w/ "NU")
./finufft1d_test$PRECSUF 1e2 2e2 $FINUFFT_REQ_TOL | sed '/NU/d'
# Checker for 1dmany routines for each type
./finufft1dmany_test$PRECSUF 3 1e2 1e3 $FINUFFT_REQ_TOL | sed '/NU/d;/T_/d' 
