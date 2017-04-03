# Makefile for Flatiron Institute (FI) NUFFT libraries.
# Barnett 4/3/17

# This is the only makefile; there are no makefiles in subdirectories.
# If you need to edit this makefile, it is recommended that you first
# copy it to makefile.local, edit that, and use make -f makefile.local

# ============= system-specific settings ===============================
CXX=g++
CC=gcc
FC=gfortran
FLINK=-lstdc++




# Notes for single-precision and/or small array size compilation:
# If you want to compile all in single-precision (saving half the RAM, but the
# speed will be only 10-20% faster), add flag -DSINGLE to CXXFLAGS.
# If you want to restrict to array sizes <2^31 and explore if 32-bit integer
# indexing beats 64-bit, add flag -DSMALLINT to CXXFLAGS which sets BIGINT
# to int.
# If you want 32 bit integers in the FINUFFT library interface instead of
# int64, add flag -DINTERFACE32 (experimental).

# Here MFLAGS are for matlab, OFLAGS for octave.
# Choose EITHER multi-threaded compile (default)...
# double-precision
#LIBSFFT = -lfftw3_threads -lfftw3 -lm
#CXXFLAGS=-fPIC -Ofast -funroll-loops -std=c++11 -fopenmp -DNEED_EXTERN_C
# single-precision
LIBSFFT = -lfftw3f_threads -lfftw3f -lm
CXXFLAGS=-fPIC -Ofast -funroll-loops -std=c++11 -fopenmp -DNEED_EXTERN_C -DSINGLE
CFLAGS=-fPIC -Ofast -funroll-loops -fopenmp
FFLAGS=-fPIC -O3 -funroll-loops -fopenmp
MFLAGS=-lgomp -largeArrayDims -lrt -D_OPENMP
# Mac users should use something like this:
#MFLAGS = -largeArrayDims -L/usr/local/gfortran/lib -lgfortran -lm -lgomp -D_OPENMP
OFLAGS=-lgomp -lrt
# for mkoctfile version >= 4.0.0 you can remove warnings by using instead:
#OFLAGS=-lgomp -lrt -std=c++11

# OR uncomment the following for single threaded compile...
# double-precision
#LIBSFFT = -lfftw3 -lm
#CXXFLAGS=-fPIC -Ofast -funroll-loops -std=c++11 -DNEED_EXTERN_C
# single-precision
#LIBSFFT = -lfftw3f -lm
#CXXFLAGS=-fPIC -Ofast -funroll-loops -std=c++11 -DNEED_EXTERN_C -DSINGLE
#CFLAGS=-fPIC -Ofast -funroll-loops
#FFLAGS=-fPIC -O3 -funroll-loops
#MFLAGS=-largeArrayDims -lrt
# Mac users should use something like this:
#MFLAGS = -largeArrayDims -L/usr/local/gfortran/lib -lgfortran -lm
#OFLAGS=-lrt -std=c++11

# Other MATLAB wrapper stuff...
MEX=mex
# Mac users should use something like this:
#MEX = /Applications/MATLAB_R2017a.app/bin/mex
# location of your MWrap executable (see INSTALL.md):
MWRAP=mwrap
# ======================================================================


# objects to compile: spreader...
SOBJS = src/cnufftspread.o src/utils.o
# for NUFFT library and its testers...
OBJS = $(SOBJS) src/finufft1d.o src/finufft2d.o src/finufft3d.o src/dirft1d.o src/dirft2d.o src/dirft3d.o src/common.o contrib/legendre_rule_fast.o src/finufft_c.o fortran/finufft_f.o
OBJS1 = $(SOBJS) src/finufft1d.o src/dirft1d.o src/common.o contrib/legendre_rule_fast.o
OBJS2 = $(SOBJS) src/finufft2d.o src/dirft2d.o src/common.o contrib/legendre_rule_fast.o
OBJS3 = $(SOBJS) src/finufft3d.o src/dirft3d.o src/common.o contrib/legendre_rule_fast.o
# for Fortran interface demos...
FOBJS = fortran/dirft1d.o fortran/dirft2d.o fortran/dirft3d.o fortran/prini.o

HEADERS = src/cnufftspread.h src/finufft.h src/dirft.h src/common.h src/utils.h src/finufft_c.h fortran/finufft_f.h

.PHONY: usage lib examples test perftest fortran matlab octave python all

default: usage

all: test perftest lib examples fortran matlab octave python

usage:
	@echo "makefile for FINUFFT library. Specify what to make:"
	@echo " make lib - compile the main libraries (in lib/)"
	@echo " make examples - compile and run codes in examples/"
	@echo " make test - compile and run math validation tests"
	@echo " make perftest - compile and run performance tests"
	@echo " make fortran - compile and test Fortran interfaces"
	@echo " make matlab - compile Matlab interfaces"
	@echo " make octave - compile and test octave interfaces"
	@echo " make python - compile and test python interfaces"
	@echo " make all - do all of the above"
	@echo " make clean - remove all object and executable files apart from MEX"
	@echo "For faster (multicore) making you may want to append the flag -j"

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
	$(CXX) -shared $(OBJS) -o lib/libfinufft.so      # fails in mac osx
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
test: lib/libfinufft.a test/testutils test/finufft1d_test test/finufft2d_test test/finufft3d_test test/dumbinputs
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
	$(FC) $(FFLAGS) fortran/nufft1d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o fortran/nufft1d_demo
	$(FC) $(FFLAGS) fortran/nufft2d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o fortran/nufft2d_demo
	$(FC) $(FFLAGS) fortran/nufft3d_demo.f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o fortran/nufft3d_demo
	time -p fortran/nufft1d_demo
	time -p fortran/nufft2d_demo
	time -p fortran/nufft3d_demo

# matlab .mex* executable...
matlab: lib/libfinufft.a $(HEADERS) matlab/finufft_m.o matlab/finufft.cpp
	$(MEX) matlab/finufft.cpp lib/libfinufft.a matlab/finufft_m.o $(MFLAGS) $(LIBSFFT) -output matlab/finufft

# octave .mex executable...
octave: lib/libfinufft.a $(HEADERS) matlab/finufft_m.o matlab/finufft.cpp
	mkoctfile --mex matlab/finufft.cpp lib/libfinufft.a matlab/finufft_m.o $(OFLAGS) $(LIBSFFT) -output matlab/finufft
	@echo "Running octave interface test; please wait a few seconds..."
	(cd matlab; octave check_finufft.m)

# rebuilds fresh MEX (matlab/octave) gateway via mwrap... (needs mwrap)
mex: matlab/finufft.cpp
matlab/finufft.cpp: matlab/finufft.mw
	(cd matlab;\
	$(MWRAP) -list -mex finufft -cppcomplex -mb finufft.mw ;\
	$(MWRAP) -mex finufft -c finufft.cpp -cppcomplex finufft.mw )

# python wrapper... (awaiting completion)
python: python/setup.py python/demo.py python/finufft/build.py python/finufft/__init__.py python/finufft/interface.cpp
	(cd python; rm -rf build ;\
	python setup.py build_ext --inplace ;\
	python demo.py )

# various obscure devel tests...
devel/plotkernels: $(SOBJS) $(HEADERS) devel/plotkernels.cpp
	$(CXX) $(CXXFLAGS) devel/plotkernels.cpp -o devel/plotkernels $(SOBJS) 
	(cd devel; ./plotkernels > plotkernels.dat)

devel/testi0: devel/testi0.cpp devel/besseli.o src/utils.o
	$(CXX) $(CXXFLAGS) devel/testi0.cpp $(OBJS) -o devel/testi0
	(cd devel; ./testi0)

# cleaning up...
clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJS3) $(FOBJS) $(SOBJS)
	rm -f test/spreadtestnd test/finufft?d_test test/testutils test/results/*.out fortran/nufft?d_demo examples/example1d1 examples/example1d1c matlab/*.o

# only do this if you have mwrap to rebuild the interfaces...
mexclean:
	rm -f matlab/finufft.cpp matlab/finufft?d?.m matlab/finufft.mex*
