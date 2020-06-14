#!/bin/bash
# a basic set of performance tests for Flatiron Institute NUFFT.
# Barnett 2/2/17, tidied 3/13/17. removed sort switch 6/13/20.

M=1e6       # problem size (sets both # NU pts and # U modes); it's a string
TOL=1e-6    # overall requested accuracy, also a string
DEBUG=0     # whether to see timing breakdowns

echo "nuffttestnd output:"
./mycpuinfo.sh

echo
echo "size = $M, tol = $TOL: multi-core tests..."
# currently we run 1e6 modes in each case, in non-equal dims (more generic):
./finufft1d_test 1e6 $M $TOL $DEBUG
./finufft2d_test 500 2000 $M $TOL $DEBUG
./finufft3d_test 100 200 50 $M $TOL $DEBUG

echo
echo "size = $M, tol = $TOL: single core tests..."
export OMP_NUM_THREADS=1
./finufft1d_test 1e6 $M $TOL $DEBUG
./finufft2d_test 500 2000 $M $TOL $DEBUG
./finufft3d_test 100 200 50 $M $TOL $DEBUG
