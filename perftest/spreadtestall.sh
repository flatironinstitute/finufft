#!/bin/bash
# tester for spreadinterp component that hits as many options & code lines
# as possible. Takes around 2-3 seconds wall-clock time.
# No math test is done (human could check "rel err" outputs small),
# since this is based on a speed-testing executable (spreadtestnd).
# For pass-fail math tests instead see ../test/finufft*test
# Barnett 10/23/20

M=1e6       # problem size (# NU pts)
N=1e6       # num U grid pts

# one thread per core (ie no hyperthreading)...
export OMP_NUM_THREADS=$(./mynumcores.sh)

echo "spreadtestall.sh :"
echo ""
echo "Double-prec spread/interp tests --------------------------------------"
echo "=========== default kernel choice ============"
TOL=1e-6    # req precision
./spreadtestnd 1 $M $N $TOL
./spreadtestnd 2 $M $N $TOL
./spreadtestnd 3 $M $N $TOL
echo "=========== kerevalmeth=0 nonstandard upsampfac + debug ============"
# nonstandard upsampfac to test with the direct kernel eval (slower)...
UP=1.5
# debug output
DEB=1
./spreadtestnd 1 $M $N $TOL 2 0 $DEB 0 0 $UP
./spreadtestnd 2 $M $N $TOL 2 0 $DEB 0 0 $UP
./spreadtestnd 3 $M $N $TOL 2 0 $DEB 0 0 $UP

echo ""
echo "Single-prec spread/interp tests --------------------------------------"
echo "=========== default kernel choice ============"
TOL=1e-3    # req precision
./spreadtestndf 1 $M $N $TOL
./spreadtestndf 2 $M $N $TOL
./spreadtestndf 3 $M $N $TOL
echo "=========== kerevalmeth=0 nonstandard upsampfac + debug ============"
./spreadtestndf 1 $M $N $TOL 2 0 $DEB 0 0 $UP
./spreadtestndf 2 $M $N $TOL 2 0 $DEB 0 0 $UP
./spreadtestndf 3 $M $N $TOL 2 0 $DEB 0 0 $UP
