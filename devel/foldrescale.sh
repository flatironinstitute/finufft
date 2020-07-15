#!/bin/bash

g++-9 -O3 -ffast-math -march=native -funroll-loops -I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp
./foldrescale_perf2 1e8

g++-9 -O3 -march=native -funroll-loops -I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp
./foldrescale_perf2 1e8
