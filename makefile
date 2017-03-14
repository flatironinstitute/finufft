# Makefile for Flatiron Institute (FI) NUFFT libraries.
# Barnett 3/14/17

# This is the only makefile; there are no makefiles in subdirectories.
# If you need to edit this makefile, it is recommended that you first
# copy it to makefile.local, edit that, and use make -f makefile.local

# ============= system-specific settings ===============================
CXX=g++
CC=gcc
FC=gfortran
FLINK=-lgfortran

# choose EITHER single threaded...
#CXXFLAGS=-fPIC -Ofast -funroll-loops -march=native -std=c++11 -DNEED_EXTERN_C
#FLAGS=-fPIC -Ofast -funroll-loops -march=native
#CLIBSFFT = -lfftw3 -lm
#FFLAGS=-fPIC -O3 -funroll-loops
#MFLAGS=

# ...OR multi-threaded:
LIBSFFT = -lfftw3_omp -lfftw3 -lm
CXXFLAGS=-fPIC -Ofast -funroll-loops -march=native -std=c++11 -fopenmp -DNEED_EXTERN_C
CFLAGS=-fPIC -Ofast -funroll-loops -march=native -fopenmp
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
OBJS = $(SOBJS) src/finufft1d.o src/finufft2d.o src/finufft3d.o src/dirft1d.o src/dirft2d.o src/dirft3d.o src/common.o contrib/legendre_rule_fast.o src/twopispread.o src/finufft_c.o fortran/finufft_f.o
OBJS1 = $(SOBJS) src/finufft1d.o src/dirft1d.o src/common.o contrib/legendre_rule_fast.o src/twopispread.o
OBJS2 = $(SOBJS) src/finufft2d.o src/dirft2d.o src/common.o contrib/legendre_rule_fast.o src/twopispread.o
OBJS3 = $(SOBJS) src/finufft3d.o src/dirft3d.o src/common.o contrib/legendre_rule_fast.o src/twopispread.o
# for Fortran interface demos...
FOBJS = fortran/dirft1d.o fortran/dirft2d.o fortran/dirft3d.o fortran/prini.o

HEADERS = src/cnufftspread.h src/finufft.h src/twopispread.h src/dirft.h src/common.h src/utils.h src/finufft_c.h fortran/finufft_f.h

default: usage

.PHONY: usage lib examples test perftest fortran

usage:
	@echo "makefile for FINUFFT library. Specify what to make:"
	@echo " make lib - compile the main libraries (in lib/)"
	@echo " make examples - compile and run codes in examples/"
	@echo " make test - compile and run validation tests"
	@echo " make perftest - compile and run performance tests"
	@echo " make fortran - compile and test Fortran interfaces"
	@echo " make clean - remove all object and executable files apart from MEX"
	@echo "For faster making you may want to append, eg, the flag -j8"

# implicit rules for objects (note -o ensures writes to correct dir)
%.o: %.cpp %.h
	$(CXX) -c $(CXXFLAGS) $< -o $@
%.o: %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@
%.o: %.f %.h
	$(FC) -c $(FFLAGS) $< -o $@

# build the library...
lib: lib/libfinufft.a lib/libfinufft.so
	echo "lib/libfinufft.a and lib/libfinufft.so built"
lib/libfinufft.a: $(OBJS) $(HEADERS)
	ar rcs lib/libfinufft.a $(OBJS)
lib/libfinufft.so: $(OBJS) $(HEADERS)
	$(CXX) -shared $(OBJS) -o lib/libfinufft.so
# see: http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html

# examples...
examples: examples/example1d1 examples/example1d1c
	examples/example1d1
	examples/example1d1c
examples/example1d1: examples/example1d1.o lib/libfinufft.a
	$(CXX) $(CXXFLAGS) examples/example1d1.o lib/libfinufft.a $(LIBSFFT) -o examples/example1d1
examples/example1d1c: examples/example1d1c.o lib/libfinufft.a
	$(CXX) $(CFLAGS) examples/example1d1c.o lib/libfinufft.a $(LIBSFFT) -o examples/example1d1c

# validation tests... (most link to .o allowing testing pieces separately)
test: lib test/testutils test/finufft1d_test test/finufft2d_test test/finufft3d_test test/dumbinputs
	(cd test; ./check_finufft.sh)
test/testutils: test/testutils.cpp src/utils.o src/utils.h $(HEADERS)
	$(CXX) $(CXXFLAGS) test/testutils.cpp src/utils.o -o test/testutils
test/finufft1d_test: test/finufft1d_test.cpp $(OBJS1) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft1d_test.cpp $(OBJS1) $(LIBSFFT) -o test/finufft1d_test
test/finufft2d_test: test/finufft2d_test.cpp $(OBJS2) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft2d_test.cpp $(OBJS2) $(LIBSFFT) -o test/finufft2d_test
test/finufft3d_test: test/finufft3d_test.cpp $(OBJS3) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft3d_test.cpp $(OBJS3) $(LIBSFFT) -o test/finufft3d_test
test/dumbinputs: test/dumbinputs.cpp lib/libfinufft.a $(HEADERS)
	$(CXX) $(CXXFLAGS) test/dumbinputs.cpp lib/libfinufft.a $(LIBSFFT) -o test/dumbinputs

# performance tests...
perftest: test/spreadtestnd test/finufft1d_test test/finufft2d_test test/finufft3d_test
# here the tee cmd copies output to screen. 2>&1 grabs both stdout and stderr...
	(cd test; ./spreadtestnd.sh 2>&1 | tee results/spreadtestnd_results.txt)
	(cd test; ./nuffttestnd.sh 2>&1 | tee results/nuffttestnd_results.txt)
test/spreadtestnd: test/spreadtestnd.cpp $(SOBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/spreadtestnd.cpp $(SOBJS) -o test/spreadtestnd

# fortran interface...
fortran: $(FOBJS) $(OBJS) $(HEADERS)
# note that linking opts seem to need to go at the end of the compile cmd:
	$(CXX) $(FFLAGS) fortran/nufft1d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) -o fortran/nufft1d_demo $(FLINK)
	$(CXX) $(FFLAGS) fortran/nufft2d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) -o fortran/nufft2d_demo $(FLINK)
	$(CXX) $(FFLAGS) fortran/nufft3d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) -o fortran/nufft3d_demo $(FLINK)
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
devel/plotkernels: $(SOBJS) $(HEADERS) devel/plotkernels.cpp
	$(CXX) $(CXXFLAGS) devel/plotkernels.cpp -o devel/plotkernels $(SOBJS) 
	(cd devel; ./plotkernels > plotkernels.dat)

devel/testi0: devel/testi0.cpp devel/besseli.o src/utils.o
	$(CXX) $(CXXFLAGS) devel/testi0.cpp $(OBJS) -o devel/testi0
	(cd devel; ./testi0)

clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJS3) $(FOBJS) $(SOBJS)
	rm -f test/spreadtestnd test/finufft?d_test test/testutils test/results/*.out fortran/nufft?d_demo examples/example1d1 examples/example1d1c
