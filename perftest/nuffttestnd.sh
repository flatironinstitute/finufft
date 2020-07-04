#!/bin/bash
# A basic set of performance tests for Flatiron Institute NUFFT.
# Relies on test executables in ../test

# Barnett 2/2/17, tidied 3/13/17. no sort 6/13/20. prec switch 7/2/20.

M=1e6       # problem size (sets both # NU pts and # U modes); it's a string
TOL=1e-6    # overall requested accuracy, also a string
DEBUG=0     # whether to see timing breakdowns

echo "nuffttestnd output:"
./mycpuinfo.sh

if [[ $1 == "SINGLE" ]]; then
    PREC=single
    PRECSUF=f
else
    PREC=double
    PRECSUF=
fi

echo
echo "$PREC-precision multi-thread tests: size = $M, tol = $TOL..."
# currently we run 1e6 modes in each case, in non-equal dims (more generic):
../test/finufft1d_test$PRECSUF 1e6 $M $TOL $DEBUG
../test/finufft2d_test$PRECSUF 500 2000 $M $TOL $DEBUG
../test/finufft3d_test$PRECSUF 100 200 50 $M $TOL $DEBUG

echo
echo "$PREC-precision single-thread tests: size = $M, tol = $TOL..."
export OMP_NUM_THREADS=1
../test/finufft1d_test$PRECSUF 1e6 $M $TOL $DEBUG
../test/finufft2d_test$PRECSUF 500 2000 $M $TOL $DEBUG
../test/finufft3d_test$PRECSUF 100 200 50 $M $TOL $DEBUG
