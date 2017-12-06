#!/bin/bash
# a basic set of performance tests for Flatiron Institute NUFFT.
# Barnett 2/2/17, tidied 3/13/17

M=1e6       # problem size (sets both # NU pts and # U modes)
TOL=1e-6    # overall requested accuracy
DEBUG=0     # whether to see timing breakdowns
SORT=1      # i7 better if sort
if grep -q Xeon /proc/cpuinfo; then
    echo "Xeon detected, switching off spreader sorting..."
    SORT=1      # whether to sort (1 also for xeon)
fi

echo "nuffttestnd output:"
./mycpuinfo.sh

echo
echo "size = $M, tol = $TOL: multi-core tests..."
# currently we run 1e6 modes in each case, in non-equal dims (more generic):
./finufft1d_test 1e6 $M $TOL $DEBUG $SORT
./finufft2d_test 500 2000 $M $TOL $DEBUG $SORT
./finufft3d_test 100 200 50 $M $TOL $DEBUG $SORT

echo
echo "size = $M, tol = $TOL: single core tests..."
export OMP_NUM_THREADS=1
./finufft1d_test 1e6 $M $TOL $DEBUG $SORT
./finufft2d_test 500 2000 $M $TOL $DEBUG $SORT
./finufft3d_test 100 200 50 $M $TOL $DEBUG $SORT
