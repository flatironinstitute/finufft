#!/bin/bash
# basic perf test of spread/interp for 2/3d, single/double
# Barnett 1/29/21, some 1D added 12/2/21.

BINDIR="./"

n=1000000
M=1000000
dist=0         # 0= random unif, 1 = clustered
Msub=10000   # claimed default is 65536
tols=1e-5
told=1e-12

echo "spread 1D.............................................."
$BINDIR/spread1d_test    1 $dist $n $Msub $M $told
$BINDIR/spread1d_test    2 $dist $n $Msub $M $told
$BINDIR/spread1d_testf 1 $dist $n $Msub $M $tols
$BINDIR/spread1d_testf 2 $dist $n $Msub $M $tols

echo "interp 1D.............................................."
$BINDIR/interp1d_test    1 $dist $n $M $told
$BINDIR/interp1d_testf 1 $dist $n $M $tols
# note there is no meth=2 in 1D interp

# 2D params... (n is grid size per dim)
n=1000
M=1000000

echo "spread 2D.............................................."
$BINDIR/spread2d_test    1 $dist $n $n $Msub $M $told
$BINDIR/spread2d_test    2 $dist $n $n $Msub $M $told
$BINDIR/spread2d_testf 1 $dist $n $n $Msub $M $tols
$BINDIR/spread2d_testf 2 $dist $n $n $Msub $M $tols

echo "interp 2D.............................................."
$BINDIR/interp2d_test    1 $dist $n $n $M $told
$BINDIR/interp2d_test    2 $dist $n $n $M $told
$BINDIR/interp2d_testf 1 $dist $n $n $M $tols
$BINDIR/interp2d_testf 2 $dist $n $n $M $tols


# 3D params...
n=100
M=1000000

echo "spread 3D.............................................."
$BINDIR/spread3d_test    1 $dist $n $n $n $Msub $M $told
# note absence of meth=2 for 3D double
$BINDIR/spread3d_testf 1 $dist $n $n $n $Msub $M $tols
$BINDIR/spread3d_testf 2 $dist $n $n $n $Msub $M $tols

echo "interp 3D.............................................."
$BINDIR/interp3d_test    1 $dist $n $n $n $M $told
# note absence of meth=2 for 3D double
$BINDIR/interp3d_testf 1 $dist $n $n $n $M $tols
$BINDIR/interp3d_testf 2 $dist $n $n $n $M $tols
