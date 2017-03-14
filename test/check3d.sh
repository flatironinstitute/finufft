#!/bin/bash
# Standard checker for all 1d routines. Sed removes the timing lines (w/ "NU")
./finufft3d_test 5 10 20 1e3 $FINUFFT_REQ_TOL 0 | sed '/NU/d'
