#!/bin/bash
# simple driver for Marco's sweeping-w spreadtest variant, all precs & dims.
# used by the makefile.
# all avail threads for now.
# human has to check the output for now.
# Barnett 6/4/24. 1/12/26: This is rather obsolete and unused.

M=1e6       # problem size (sets both # NU pts and # U modes); it's a string
N=1e6       # num U grid pts

./spreadtestndall 1 $M $N 1
./spreadtestndall 1 $M $N 2
./spreadtestndall 2 $M $N 1
./spreadtestndall 2 $M $N 2
./spreadtestndall 3 $M $N 1
./spreadtestndall 3 $M $N 2
./spreadtestndallf 1 $M $N 1
./spreadtestndallf 1 $M $N 2
./spreadtestndallf 2 $M $N 1
./spreadtestndallf 2 $M $N 2
./spreadtestndallf 3 $M $N 1
./spreadtestndallf 3 $M $N 2
