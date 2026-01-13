#!/bin/bash
# simple driver for Marco's sweeping-w spreadtest variant, all precs & dims.
# used by the makefile.
# all avail threads for now.
# human has to check the output for now.
# Barnett 6/4/24.
# Barnett 1/12/26 update for new cmd line args.

M=1e6       # problem size (sets both # NU pts and # U modes); it's a string
N=1e6       # num U grid pts
USF=2.0     # sigma upsampfac (it's also a string)

echo ""
echo "Double-prec spread/interp tol sweep ----------------------------------"
echo ""
./spreadtestndall  1 $M $N 2 $USF
echo ""
./spreadtestndall  2 $M $N 2 $USF
echo ""
./spreadtestndall  3 $M $N 2 $USF
echo ""
echo "Single-prec spread/interp tol sweep ----------------------------------"
echo ""
./spreadtestndallf 1 $M $N 2 $USF
echo ""
./spreadtestndallf 2 $M $N 2 $USF
echo ""
./spreadtestndallf 3 $M $N 2 $USF
