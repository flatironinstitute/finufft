#!/bin/bash
# a basic set of multidimensional spreader tests. Barnett 1/16/17

# show how many threads of what CPU
grep "model name" /proc/cpuinfo

echo
echo "multi-core..."
./spreadtestnd 1 1e-6
./spreadtestnd 2 1e-6
./spreadtestnd 3 1e-6

echo
echo "single core..."
# I don't understand why export needed here to have effect in script
export OMP_NUM_THREADS=1
./spreadtestnd 1 1e-6
./spreadtestnd 2 1e-6
./spreadtestnd 3 1e-6
