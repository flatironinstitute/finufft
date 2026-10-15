#!/bin/bash
# A basic set of performance tests for Flatiron Institute NUFFT.
# Relies on test executables in ../test

# Barnett 2/2/17, tidied 3/13/17. no sort 6/13/20. prec switch, threads 7/2/20.

M=1e6   # problem size (sets both # NU pts and # U modes); it's a string
DEBUG=0 # whether to see timing breakdowns

#TESTTHREADS=$(./mymaxthreads.sh)      # max threads (hyperthreading)
TESTTHREADS=$(./mynumcores.sh) # one thread per core (no hyperthreading)

echo "nuffttestnd output:"
./mycpuinfo.sh

if [[ $1 == "SINGLE" ]]; then
	PREC=single
	PRECSUF=f
	TOL=1e-3 # requested accuracy, also a string (tweaked to avoid ier=26)
else
	PREC=double
	PRECSUF=
	TOL=1e-6 # requested accuracy, also a string
fi

echo
export OMP_NUM_THREADS=$TESTTHREADS
echo "$PREC-precision $OMP_NUM_THREADS-thread tests: size = $M, tol = $TOL..."
# 1d we limit the modes so non-reachable tol (ier=26) not triggered in SINGLE:
../test/finufft1d_test$PRECSUF 1e4 $M $TOL $DEBUG
# currently we run 1e6 modes in 2d & 3d, in non-equal dims (more generic):
../test/finufft2d_test$PRECSUF 500 2000 $M $TOL $DEBUG
../test/finufft3d_test$PRECSUF 100 200 50 $M $TOL $DEBUG

echo
export OMP_NUM_THREADS=1
echo "$PREC-precision $OMP_NUM_THREADS-thread tests: size = $M, tol = $TOL..."
../test/finufft1d_test$PRECSUF 1e4 $M $TOL $DEBUG
../test/finufft2d_test$PRECSUF 500 2000 $M $TOL $DEBUG
../test/finufft3d_test$PRECSUF 100 200 50 $M $TOL $DEBUG
