#!/bin/bash
# basic perf test of spread/interp for 2/3d, single/double
# Barnett 1/29/21, some 1D added 12/2/21.

BIN=../bin/fseries_kernel_test
DIM=1

echo "Double.............................................."
for N in 1e2 5e2 1e3 5e3 1e4 5e4 1e5 5e5
do
	for TOL in 1e-8
	do
		$BIN $N $DIM $TOL 0
		$BIN $N $DIM $TOL 1
	done
done

BIN=../bin/fseries_kernel_test_32
echo "Single.............................................."
for N in 1e2 5e2 1e3 5e3 1e4 5e4 1e5 5e5
do
	for TOL in 1e-6
	do
		$BIN $N $DIM $TOL 0
		$BIN $N $DIM $TOL 1
	done
done
