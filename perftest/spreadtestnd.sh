#!/bin/bash
# a basic quick set of quasi-uniform multidimensional spreader speed tests.
# Usage:
# double-prec:  ./spreadtestnd.sh
# single-prec:  ./spreadtestnd.sh SINGLE

# Barnett started 2/2/17. both-precision handling, choose # threads 7/3/20.

M=1e6       # problem size (# NU pts)
N=1e6       # num U grid pts
TOL=1e-6    # overall requested accuracy

#TESTTHREADS=$(./mymaxthreads.sh)      # max threads (hyperthreading)
TESTTHREADS=$(./mynumcores.sh)        # one thread per core (no hyperthreading)

echo "spreadtestnd output:"
./mycpuinfo.sh

if [[ $1 == "SINGLE" ]]; then
    PREC=single
    ST=./spreadtestndf
else
    PREC=double
    ST=./spreadtestnd
fi
   
echo
export OMP_NUM_THREADS=$TESTTHREADS
echo "$PREC-precision $OMP_NUM_THREADS-thread tests: #NU = $M, #U = $N, tol = $TOL..."
$ST 1 $M $N $TOL
$ST 2 $M $N $TOL
$ST 3 $M $N $TOL

echo
export OMP_NUM_THREADS=1
echo "$PREC-precision $OMP_NUM_THREADS-thread tests: #NU = $M, #U = $N, tol = $TOL..."
$ST 1 $M $N $TOL
$ST 2 $M $N $TOL
$ST 3 $M $N $TOL
