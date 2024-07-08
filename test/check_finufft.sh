#!/bin/bash

# Validation tests for FINUFFT library. On Windows, we append .exe to the names of the executables.
# Usage:   To do double-precision tests:   ./check_finufft.sh
#          To do single-precision tests:   ./check_finufft.sh SINGLE
#          To run test on Windows (WSL):   ./check_finufft.sh {DUMMY or SINGLE} ON
# Exit code is 0 for success, otherwise failure

# These are supported as of v2.2.0, but will become obsolete in favor of CTest.

# In total these tests take about 5 seconds on a modern machine with the
# default compile, threads=2*#cores (hyperthreading), or <1 second w/ less thr.
# (This sounds backwards, but is true; believed OMP overhead in FINUFFT calls.)

# Barnett 3/14/17. numdiff-free option 3/16/17. simpler, dual-prec 7/3/20,
# execs now have exit codes, removed any numdiff dep 8/18/20
# removed diff 6/16/23. Added kerevalmeth=0 vs 1 test 7/8/24.

# precision-specific settings
if [[ $1 == "SINGLE" ]]; then
    PREC=single
    # what's weird is that tol=1e-6 here gives *worse* single-prec errs >2e-4 :(
    export FINUFFT_REQ_TOL=1e-5
    # acceptable error one digit above requested tol... (& rounding accum)
    CHECK_TOL=2e-4
    # modifier for executables, exported so that check?d.sh can also access...
    export PRECSUF=f
else
    PREC=double
    export FINUFFT_REQ_TOL=1e-12
    CHECK_TOL=1e-11
    export PRECSUF=
fi
if [[ $2 == "ON" ]]; then
    export FEX=".exe"
else
    export FEX=
fi
# Note that bash cannot handle floating-point arithmetic, and bc cannot read
# exponent notation. Thus the exponent notation above is purely string in nature

SIGSEGV=139            # POSIX code to catch a seg violation: 128 + 11
CRASHES=0
FAILS=0
N=0
DIR=results
echo "pass-fail FINUFFT library $PREC-precision check with tol=$FINUFFT_REQ_TOL ..."

# no loop, just do one test after another (simpler, less abstraction)
# Note: prec-dep results files are written in DIR
# TESTS -------------------------------------------------------------

((N++))
T=testutils$PRECSUF
# stdout to screen and file; stderr to different file
./$T$FEX 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}          # exit code of the tested cmd (not the tee cmd!)
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

((N++))
T=finufft1d_test$PRECSUF
./$T$FEX 1e2 2e2 $FINUFFT_REQ_TOL 0 2 0.0 $CHECK_TOL 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

((N++))
T=finufft1dmany_test$PRECSUF
./$T$FEX 3 1e2 1e3 $FINUFFT_REQ_TOL 0 0 0 2 0.0 $CHECK_TOL 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

((N++))
T=finufft2d_test$PRECSUF
./$T$FEX 1e2 1e1 1e3 $FINUFFT_REQ_TOL 0 2 0.0 $CHECK_TOL 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

((N++))
T=finufft2dmany_test$PRECSUF
./$T$FEX 3 1e2 1e1 1e3 $FINUFFT_REQ_TOL 0 0 0 2 0.0 $CHECK_TOL 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

((N++))
T=finufft3d_test$PRECSUF
./$T$FEX 5 10 20 1e2 $FINUFFT_REQ_TOL 0 2 0.0 $CHECK_TOL 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

((N++))
T=finufft3dmany_test$PRECSUF
./$T$FEX 2 10 50 20 1e2 $FINUFFT_REQ_TOL 0 0 0 2 0.0 $CHECK_TOL 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

((N++))
T=finufft3dkernel_test$PRECSUF
./$T$FEX 20 50 30 1e3 $FINUFFT_REQ_TOL 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

((N++))
T=dumbinputs$PRECSUF
./$T$FEX 2>$DIR/$T.err.out | tee $DIR/$T.out
E=${PIPESTATUS[0]}
if [[ $E -eq 0 ]]; then echo passed; elif [[ $E -eq $SIGSEGV ]]; then echo crashed; ((CRASHES++)); else echo failed; ((FAILS++)); fi

# END TESTS ---------------------------------------------------------


echo "check_finufft.sh $PREC-precision done. Summary:"
echo "$CRASHES segfaults out of $N tests done"
echo "$FAILS fails out of $N tests done"
echo ""
exit $((CRASHES+FAILS))         # use total as exit code
