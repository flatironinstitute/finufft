#!/bin/bash
# a basic set of multidimensional spreader tests.
# Barnett 2/2/17, 3/13/17. M,N indep 3/27/17

M=1e6       # problem size (# NU pts)
N=1e6       # num U grid pts
TOL=1e-6    # overall requested accuracy

echo "spreadtestnd output:"
./mycpuinfo.sh

echo
echo "#NU = $M, #U = $N, tol = $TOL: multi-core tests..."
./spreadtestnd 1 $M $N $TOL
./spreadtestnd 2 $M $N $TOL
./spreadtestnd 3 $M $N $TOL

echo
echo "#NU = $M, #U = $N, tol = $TOL: single core tests..."
export OMP_NUM_THREADS=1
./spreadtestnd 1 $M $N $TOL
./spreadtestnd 2 $M $N $TOL
./spreadtestnd 3 $M $N $TOL
