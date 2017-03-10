# Makefile for Flatiron Institute (FI) NUFFT libraries.
# Barnett 3/10/17

# This is the only makefile; there are no makefiles in subdirectories.
# If you need to edit this makefile, it is recommended that you first
# copy it to makefile.local, edit that, and use make -f makefile.local

# ============= system-specific settings ===============================
CC=g++
FC=gfortran
FLINK=-lgfortran

# choose EITHER single threaded...
#CFLAGS=-fPIC -Ofast -funroll-loops -march=native -std=c++11
#CLIBSFFT = -lfftw3 -lm
#FFLAGS=-fPIC -O3 -funroll-loops
#MFLAGS=

# ...OR multi-threaded:
CFLAGS=-fPIC -Ofast -funroll-loops -march=native -std=c++11 -fopenmp
LIBSFFT = -lfftw3_omp -lfftw3 -lm
FFLAGS=-fPIC -O3 -funroll-loops -fopenmp
MFLAGS=-lgomp

# MAC OSX to do...

# MATLAB stuff..
MEX=mex
MEXFLAGS = -largeArrayDims -lgfortran -lm $(MFLAGS)
# Mac users should use something like this:
#MEX = /Applications/MATLAB_R2014a.app/bin/mex
#MEXFLAGS = -largeArrayDims -L/usr/local/gfortran/lib -lgfortran -lm
# ======================================================================


# objects to compile: spreader...
SOBJS = src/cnufftspread.o src/utils.o
# for NUFFT library and its testers...
OBJS = $(SOBJS) src/finufft1d.o src/finufft2d.o src/finufft3d.o src/dirft1d.o src/dirft2d.o src/dirft3d.o src/common.o contrib/legendre_rule_fast.o src/twopispread.o fortran/finufft_f.o
OBJS1 = $(SOBJS) src/finufft1d.o src/dirft1d.o src/common.o contrib/legendre_rule_fast.o src/twopispread.o
OBJS2 = $(SOBJS) src/finufft2d.o src/dirft2d.o src/common.o contrib/legendre_rule_fast.o src/twopispread.o
OBJS3 = $(SOBJS) src/finufft3d.o src/dirft3d.o src/common.o contrib/legendre_rule_fast.o src/twopispread.o
# for Fortran interface and demos...
FOBJS = fortran/finufft_f.o fortran/dirft1d.o fortran/dirft2d.o fortran/dirft3d.o fortran/prini.o

HEADERS = src/cnufftspread.h src/finufft.h src/twopispread.h src/dirft.h src/common.h src/utils.h

default: lib

.PHONY: lib test test1d test2d test3d testnd fortran

# implicit rules for objects (note -o ensures writes to correct dir)
.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@
.c.o:
	$(CC) -c $(CFLAGS) $< -o $@
.f.o:
	$(FC) -c $(FFLAGS) $< -o $@

# build libraries...
lib: lib/libfinufft.a lib/libfinufft.so
lib/libfinufft.a: $(OBJS)
	ar rcs lib/libfinufft.a $(OBJS)
lib/libfinufft.so: $(OBJS)
	$(CC) -shared -Wl,-soname,lib/libfinufft.so.0 -o lib/libfinufft.so.0.7
# see: http://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html

# validation tests...
test: test/finufft1d_test test/finufft2d_test test/finufft3d_test
	(cd test; ./check_finufft.sh)

test/finufft1d_test: test/finufft1d_test.cpp $(OBJS1) $(HEADERS)
	$(CC) $(CFLAGS) test/finufft1d_test.cpp $(OBJS1) $(LIBSFFT) -o test/finufft1d_test

test/finufft2d_test: test/finufft2d_test.cpp $(OBJS2) $(HEADERS)
	$(CC) $(CFLAGS) test/finufft2d_test.cpp $(OBJS2) $(LIBSFFT) -o test/finufft2d_test

test/finufft3d_test: test/finufft3d_test.cpp $(OBJS3) $(HEADERS)
	$(CC) $(CFLAGS) test/finufft3d_test.cpp $(OBJS3) $(LIBSFFT) -o test/finufft3d_test



# performance tests...
perftest: testnd



# test drivers and scripts...
test/spreadtestnd: test/spreadtestnd.cpp $(SOBJS) $(HEADERS)
	$(CC) $(CFLAGS) test/spreadtestnd.cpp $(SOBJS) -o test/spreadtestnd

spreadtestnd: test/spreadtestnd
# here the tee cmd copies output to screen. 2>&1 grabs both stdout and stderr...
	(cd test; ./spreadtestnd.sh 2>&1 | tee results/spreadtestnd_results.txt)

test1d: test/finufft1d_test
	test/finufft1d_test 1e4 1e3 1e-6 1         # small prob for accuracy

test2d: test/finufft2d_test
	test/finufft2d_test 200 50 1e3 1e-6 1      # small

test3d: test/finufft3d_test
	test/finufft3d_test 20 100 50 1e2 1e-6 1   # small

testnd: test/finufft1d_test test/finufft2d_test test/finufft3d_test
	(cd test; ./nuffttestnd.sh 2>&1 | tee results/nuffttestnd_results.txt)

fortran: $(FOBJS) $(OBJS) $(HEADERS)
# note that linking opts seem to need to go at the end of the compile cmd:
	$(CC) $(FFLAGS) fortran/nufft1d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) -o fortran/nufft1d_demo $(FLINK)
	$(CC) $(FFLAGS) fortran/nufft2d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) -o fortran/nufft2d_demo $(FLINK)
	$(CC) $(FFLAGS) fortran/nufft3d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) -o fortran/nufft3d_demo $(FLINK)
	time -p fortran/nufft1d_demo
	time -p fortran/nufft2d_demo
	time -p fortran/nufft3d_demo

# todo: make mex interfaces...
#mex: src/cnufftspread.h src/_mcwrap/mcwrap_cnufftspread_dir1.cpp $(SPREADOBJS)
# make new interface in matlab: from src/, do mcwrap('cnufftspread.h')
# which fails for omp.
# mv src/cnufftspread_type1.mexa64 matlab/
#	(cd src; $(MEX) _mcwrap/mcwrap_cnufftspread_type1.cpp cnufftspread.o ../contrib/besseli.o -output cnufftspread_type1 $(MEXFLAGS))

# todo: python wrapper...

# various obscure testers (experts only)...
test/testutils: test/testutils.cpp src/utils.o src/utils.h
	$(CC) $(CFLAGS) test/testutils.cpp src/utils.o -o test/testutils
	(cd test; ./testutils)

devel/plotkernels: $(SOBJS) $(HEADERS) devel/plotkernels.cpp
	$(CC) $(CFLAGS) devel/plotkernels.cpp -o devel/plotkernels $(SOBJS) 
	(cd devel; ./plotkernels > plotkernels.dat)

devel/testi0: devel/testi0.cpp devel/besseli.o src/utils.o
	$(CC) $(CFLAGS) devel/testi0.cpp $(OBJS) -o devel/testi0
	(cd devel; ./testi0)

clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJS3) $(FOBJS) $(SOBJS)
	rm -f test/spreadtestnd test/finufft?d_test fortran/nufft?d_demo
