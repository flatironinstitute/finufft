#!/bin/bash
# a basic quick set of quasi-uniform multidimensional spreader speed tests.
# Barnett 2/2/17, 3/13/17. M,N indep 3/27/17, sort 3/28/17, xeon sort 11/7/17 

M=1e6       # problem size (# NU pts)
N=1e6       # num U grid pts
TOL=1e-6    # overall requested accuracy

SORT=2      # default setting for sort
if grep -q Xeon /proc/cpuinfo; then
    echo "Xeon detected"
    SORT=2      # xeon setting, also the default
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
