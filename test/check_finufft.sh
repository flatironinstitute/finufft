#!/bin/bash
# main script to loop through all validation tests for FINUFFT library.
# Barnett 3/14/17. numdiff-free option 3/16/17
# currently uses default spread_opts.sort_data 3/28/17

# Note that bash cannot handle floating-point arithmetic, and bc cannot read
# exponent notation. Thus the exponent notation here is purely string in nature:
export FINUFFT_REQ_TOL=1e-5          # requested acc, passed to check?d.sh
FINUFFT_CHECK_TOL=1e-3               # allow up to 1 digit worse then requested
DIR=results

TESTS="testutils check1d.sh check2d.sh check3d.sh dumbinputs"

if type numdiff &> /dev/null; then
    echo "numdiff appears to be installed"
    echo "pass-fail FINUFFT library check at requested accuracy $FINUFFT_REQ_TOL ..."
    CRASHES=0
    FAILS=0
    N=0
    for t in $TESTS; do
	((N++))
	echo "Test number $N: $t"
	rm -f $DIR/$t.out
	./$t | tee $DIR/$t.out              # stdout only; tee duplicates to screen
	# $? is exit code of last thing...
	if [ $? -eq 0 ]; then echo completed; else echo crashed; ((CRASHES++)); fi 
	# since refout contains 0 for each error field, relerr=1 so 2 is for safety:
	numdiff -q $DIR/$t.refout $DIR/$t.out -a $FINUFFT_CHECK_TOL -r 2.0
	if [ $? -eq 0 ]; then echo accuracy passed; else echo accuracy failed; ((FAILS++)); fi
	echo
    done
    echo "$CRASHES crashes out of $N tests done"
    echo "$FAILS fails out of $N tests done"
    exit $((CRASHES+FAILS))               # use total as exit code

else
    echo "numdiff not installed"
    echo "FINUFFT library check at requested accuracy $FINUFFT_REQ_TOL ..."
    CRASHES=0    
    N=0
    for t in $TESTS; do
	((N++))
	echo "Test number $N: $t"
	rm -f $DIR/$t.out
	./$t | tee $DIR/$t.out              # stdout only; tee duplicates to screen
	# $? is exit code of last thing...
	if [ $? -eq 0 ]; then echo completed; else echo crashed; ((CRASHES++)); fi 
	echo
    done
    echo "$CRASHES crashes out of $N tests done"
    echo "Please check by eye that the numerical output has errors at expected level!"
    echo "(or install numdiff and rerun; see ../INSTALL.md)"
    exit $((CRASHES))               # use total as exit code
fi
