#!/bin/bash

echo "BINNING (small N)...."

g++ -O3 -march=native -funroll-loops -I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp
./foldrescale_perf2

g++ -O3 -march=native -funroll-loops -I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp -ffast-math -fno-finite-math-only
./foldrescale_perf2

echo "NOBIN...."

g++ -O3 -march=native -funroll-loops -I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp -DNOBIN
./foldrescale_perf2

g++ -O3 -march=native -funroll-loops -I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp -ffast-math -fno-finite-math-only -DNOBIN
./foldrescale_perf2
