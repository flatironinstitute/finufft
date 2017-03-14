#!/bin/bash
# a basic set of multidimensional spreader tests. Barnett 2/2/17, 3/13/17

M=1e6       # problem size (sets both # NU pts and # U modes)
TOL=1e-6    # overall requested accuracy

echo "spreadtestnd output:"
./mycpuinfo.sh

echo
echo "size = $M, tol = $TOL: multi-core tests..."
# currently the spreadtestnd.cpp code fixes the test size (M,N1,...):
./spreadtestnd 1 $M $TOL
./spreadtestnd 2 $M $TOL
./spreadtestnd 3 $M $TOL

echo
echo "size = $M, tol = $TOL: single core tests..."
export OMP_NUM_THREADS=1
./spreadtestnd 1 $M $TOL
./spreadtestnd 2 $M $TOL
./spreadtestnd 3 $M $TOL
