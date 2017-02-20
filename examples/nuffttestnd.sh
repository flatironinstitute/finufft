#!/bin/bash
# a basic set of tests for Flatiron Institute NUFFT.  Barnett 2/2/17

echo "nuffttestnd output:"
./mycpuinfo.sh

export NUFFTTESTDEBUG=0     # whether to see timing breakdowns
export NUFFTTESTTOL=1e-6    # overall requested accuracy
export NUFFTTESTM=1e6       # number of nonuniform pts to test

echo
echo "tol = $NUFFTTESTTOL: multi-core tests..."
# currently we run 1e6 modes in each case, in non-equal dims (more generic):
./finufft1d_test 1e6 $NUFFTTESTM $NUFFTTESTTOL $NUFFTTESTDEBUG
./finufft2d_test 500 2000 $NUFFTTESTM $NUFFTTESTTOL $NUFFTTESTDEBUG
./finufft3d_test 100 200 50 $NUFFTTESTM $NUFFTTESTTOL $NUFFTTESTDEBUG

echo
echo "tol = $NUFFTTESTTOL: single core tests..."
export OMP_NUM_THREADS=1
./finufft1d_test 1e6 $NUFFTTESTM $NUFFTTESTTOL $NUFFTTESTDEBUG
./finufft2d_test 500 2000 $NUFFTTESTM $NUFFTTESTTOL $NUFFTTESTDEBUG
./finufft3d_test 100 200 50 $NUFFTTESTM $NUFFTTESTTOL $NUFFTTESTDEBUG
