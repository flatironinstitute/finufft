#!/bin/bash
# test range of requested accuracies, for both spreader and nufft, for a given
# single dimension.
# Usage:  ./checkallaccs.sh [dim]
# where dim = 1, 2, or 3.
# Barnett 2/17/17. Default dim=1 4/5/17

DEFAULTDIM=1
DIM=${1:-$DEFAULTDIM}
echo checkallaccs for dim=$DIM :

# finufft test size params (prod{N}.N < TEST_BIGPROB so compares direct transf)
TEST1="1e3 1e3"
TEST2="1e2 2e1 1e3"
TEST3="1e1 2e1 3e1 1e3"
# bash hack to make DIM switch between one of the above 3 choices
TESTD=TEST$DIM
TEST=${!TESTD}

# other test args
SORT=2
UPSAMPFAC=2.0

for acc in `seq 1 15`;
do
    TOL=1e-$acc
    echo ----------requesting $TOL :
#    ./spreadtestnd $DIM 1e6 1e6 $TOL $SORT
    ./finufft${DIM}d_test $TEST $TOL 0 $SORT $UPSAMPFAC | grep dirft
done
