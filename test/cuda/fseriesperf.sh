#!/bin/bash
# basic perf test of compute fseries for 1d, single/double
# Melody 02/20/22

BINDIR=./

BIN=$BINDIR/fseries_kernel_test
DIM=1

echo "Double.............................................."
for N in 1e2 5e2 1e3 2e3 5e3 1e4 5e4 1e5 5e5
do
	for TOL in 1e-8
	do
		$BIN $N $DIM $TOL 0
		$BIN $N $DIM $TOL 1
	done
done

BIN=$BINDIR/fseries_kernel_testf
echo "Single.............................................."
for N in 1e2 5e2 1e3 2e3 5e3 1e4 5e4 1e5 5e5
do
	for TOL in 1e-6
	do
		$BIN $N $DIM $TOL 0
		$BIN $N $DIM $TOL 1
	done
done
