#!/bin/bash
# a basic set of multidimensional spreader tests.
# Barnett 2/2/17, 3/13/17. M,N indep 3/27/17, sort 3/28/17

M=1e6       # problem size (# NU pts)
N=1e6       # num U grid pts
TOL=1e-6    # overall requested accuracy

SORT=1      # generally better if sort (eg i7)
if grep -q Xeon /proc/cpuinfo; then
    echo "Xeon detected, switching off spreader sorting..."
    SORT=0      # xeon is faster without sorting
fi

echo "spreadtestnd output:"
./mycpuinfo.sh

echo
echo "#NU = $M, #U = $N, tol = $TOL, sort = $SORT: multi-core tests..."
./spreadtestnd 1 $M $N $TOL $SORT
./spreadtestnd 2 $M $N $TOL $SORT
./spreadtestnd 3 $M $N $TOL $SORT

echo
echo "#NU = $M, #U = $N, tol = $TOL, sort = $SORT: single core tests..."
export OMP_NUM_THREADS=1
./spreadtestnd 1 $M $N $TOL $SORT
./spreadtestnd 2 $M $N $TOL $SORT
./spreadtestnd 3 $M $N $TOL $SORT
