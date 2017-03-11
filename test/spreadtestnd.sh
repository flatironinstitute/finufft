#!/bin/bash
# a basic set of multidimensional spreader tests. Barnett 2/2/17

echo "spreadtestnd output:"
./mycpuinfo.sh
export NUFFTTESTTOL=1e-6    # overall requested accuracy

echo
echo "tol = $NUFFTTESTTOL: multi-core tests..."
# currently the spreadtestnd.cpp code fixes the test size (M,N1,...):
./spreadtestnd 1 $NUFFTTESTTOL
./spreadtestnd 2 $NUFFTTESTTOL
./spreadtestnd 3 $NUFFTTESTTOL

echo
echo "tol = $NUFFTTESTTOL: single core tests..."
export OMP_NUM_THREADS=1
./spreadtestnd 1 $NUFFTTESTTOL
./spreadtestnd 2 $NUFFTTESTTOL
./spreadtestnd 3 $NUFFTTESTTOL
