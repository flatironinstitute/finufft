#!/bin/bash

# Main validation tests for FINUFFT library.
# To do double-precision tests:   ./check_finufft.sh
# To do single-precision tests:   ./check_finufft.sh SINGLE
# In total these tests take about 5 seconds on a modern machine.

# Also see: check?d.sh

# Barnett 3/14/17. numdiff-free option 3/16/17. simpler, dual-prec 7/3/20.

# precision-specific settings
if [[ $1 == "SINGLE" ]]; then
    PREC=single
    # what's weird is that tol=1e-6 here gives *worse* single-prec errs >2e-4 :(
    export FINUFFT_REQ_TOL=1e-5
    # acceptable error one digit above requested tol...
    CHECK_TOL=1e-4
    # modifier for executables, exported so that check?d.sh can also access...
    export PRECSUF=f
else
    PREC=double
    export FINUFFT_REQ_TOL=1e-12
    # acceptable error one digit above requested tol...
    CHECK_TOL=1e-11
    export PRECSUF=
fi
# Note that bash cannot handle floating-point arithmetic, and bc cannot read
# exponent notation. Thus the exponent notation above is purely string in nature

DIR=results
# test executable list without precision suffix (also used for .refout)...
TESTS="testutils check1d.sh check2d.sh check3d.sh dumbinputs"

if type numdiff &> /dev/null; then
    echo "numdiff appears to be installed"
    echo "pass-fail FINUFFT library $PREC-precision check with tol=$FINUFFT_REQ_TOL ..."
    CRASHES=0
    FAILS=0
    N=0
    for t in $TESTS; do
	((N++))
	echo "Test number $N: $t"
	rm -f $DIR/$t.out
        # EXEC is the executable: don't add f to .sh script names...
        if [[ $t == *.sh ]]; then
            EXEC=./$t
        else
            EXEC=./$t$PRECSUF
        fi
	$EXEC 2>$DIR/$t.err.out | tee $DIR/$t.out   # stdout only; tee duplicates to screen
	# $? is exit code of last thing...
	if [ $? -eq 0 ]; then echo completed; else echo crashed; ((CRASHES++)); fi
	# since refout contains 0 for each error field, relerr=1 so 2 is for safety:
	numdiff -q $DIR/$t.refout $DIR/$t.out -a $CHECK_TOL -r 2.0
	if [ $? -eq 0 ]; then echo accuracy passed; else echo accuracy failed; ((FAILS++)); fi
	echo
    done
    echo "check_finufft.sh $PREC-precision done:"
    echo "$CRASHES crashes out of $N tests done"
    echo "$FAILS fails out of $N tests done"
    echo ""
    exit $((CRASHES+FAILS))         # use total as exit code

else
    echo "numdiff not installed"
    echo "FINUFFT library $PREC-precision check with tol=$FINUFFT_REQ_TOL ..."
    CRASHES=0    
    N=0
    for t in $TESTS; do
	((N++))
	echo "Test number $N: $t"
	rm -f $DIR/$t.out
        if [[ $t == *.sh ]]; then
            EXEC=./$t
        else
            EXEC=./$t$PRECSUF
        fi
	$EXEC | tee $DIR/$t.out          # stdout only; tee duplicates to screen
	# $? is exit code of last thing...
	if [ $? -eq 0 ]; then echo completed; else echo crashed; ((CRASHES++)); fi 
	echo
    done
    echo "check_finufft.sh $PREC-precision done:"
    echo "$CRASHES crashes out of $N tests done"
    echo "Please check by eye that above errors do not exceed $CHECK_TOL !"
    echo "(or install numdiff and rerun; see ../docs/install.rst)"
    echo ""
    exit $((CRASHES))               # use total as exit code
fi
