# Makefile for FINUFFT.
# Barnett 2/16/18

# This is the only makefile; there are no makefiles in subdirectories.
# If users need to edit this makefile, it is recommended that they first
# copy it to makefile.local, edit that, and use make -f makefile.local

# Compilation options: (also see docs/)
#
# 1) Use "make [task] PREC=SINGLE" for single-precision, otherwise will be
#    double-precision. Single-precision saves half the RAM, and increases
#    speed slightly (<20%). Not available for matlab and octave interfaces.
# 2) make with OMP=OFF for single-threaded, otherwise multi-threaded (openmp).
# 3) Since library names for single/double and single-/multi-threaded are
#    currently shared, you cannot maintain more than one such option in the
#    same directory, and you need to make clean before switching option.
# 4) If you want to restrict to array sizes <2^31 and explore if 32-bit integer
#    indexing beats 64-bit, add flag -DSMALLINT to CXXFLAGS which sets BIGINT
#    to int.
# 5) If you want 32 bit integers in the FINUFFT library interface instead of
#    int64, add flag -DINTERFACE32 (experimental; C,F,M,O interfaces will break)

# compilers...
CXX=g++
CC=gcc
FC=gfortran
# for non-C++ compilers to be able to link to library...
CLINK=-lstdc++
FLINK=-lstdc++

# basic compile flags for single-threaded, double precision...
CXXFLAGS = -fPIC -Ofast -funroll-loops -march=native -DNEED_EXTERN_C
CFLAGS = -fPIC -Ofast -funroll-loops -march=native
FFLAGS = -fPIC -O3 -funroll-loops -march=native
# Now MFLAGS are for MATLAB MEX compilation, OFLAGS for octave mkoctfile:
MFLAGS = -largeArrayDims -lrt
# Mac users instead should use something like this:
#MFLAGS = -largeArrayDims -L/usr/local/gfortran/lib -lgfortran -lm
OFLAGS = -lrt
# location of MATLAB's mex compiler...
MEX=mex
# Mac users instead should use something like this:
#MEX = /Applications/MATLAB_R2017a.app/bin/mex
# For experts doing make mex: location of MWrap executable (see docs/install.rst):
MWRAP=mwrap

# choose the precision (sets fftw library names, test precisions)...
ifeq ($(PREC),SINGLE)
CXXFLAGS += -DSINGLE
CFLAGS += -DSINGLE
SUFFIX = f
REQ_TOL = 1e-6
CHECK_TOL = 2e-4
else
SUFFIX =
REQ_TOL = 1e-12
CHECK_TOL = 1e-11
CXXFLAGS += -DVECT  # Interpolation has explicit vectorization only in dbl prec
CFLAGS += -DVECT
endif
FFTW = fftw3$(SUFFIX)
LIBSFFT = -l$(FFTW) -lm

# multi-threaded libs & flags needed...
ifneq ($(OMP),OFF)
CXXFLAGS += -fopenmp
CFLAGS += -fopenmp
FFLAGS += -fopenmp
MFLAGS += -lgomp -D_OPENMP
OFLAGS += -lgomp
LIBSFFT += -l$(FFTW)_threads
endif

# ======================================================================

# objects to compile: spreader...
SOBJS = src/cnufftspread.o src/utils.o
# for NUFFT library and its testers...
OBJS = $(SOBJS) src/finufft1d.o src/finufft2d.o src/finufft3d.o src/dirft1d.o src/dirft2d.o src/dirft3d.o src/common.o contrib/legendre_rule_fast.o src/finufft_c.o fortran/finufft_f.o
OBJS1 = $(SOBJS) src/finufft1d.o src/dirft1d.o src/common.o contrib/legendre_rule_fast.o
OBJS2 = $(SOBJS) src/finufft2d.o src/dirft2d.o src/common.o contrib/legendre_rule_fast.o
OBJS3 = $(SOBJS) src/finufft3d.o src/dirft3d.o src/common.o contrib/legendre_rule_fast.o
# for Fortran interface demos...
FOBJS = fortran/dirft1d.o fortran/dirft2d.o fortran/dirft3d.o fortran/dirft1df.o fortran/dirft2df.o fortran/dirft3df.o fortran/prini.o

HEADERS = src/cnufftspread.h src/finufft.h src/dirft.h src/common.h src/utils.h src/finufft_c.h fortran/finufft_f.h

.PHONY: usage lib examples test perftest fortran matlab octave all mex

default: usage

all: test perftest lib examples fortran matlab octave

usage:
	@echo "Makefile for FINUFFT library. Specify what to make:"
	@echo " make lib - compile the main libraries (in lib/ and lib-static/)"
	@echo " make examples - compile and run codes in examples/"
	@echo " make test - compile and run math validation tests"
	@echo " make perftest - compile and run performance tests"
	@echo " make fortran - compile and test Fortran interfaces"
	@echo " make matlab - compile Matlab interfaces"
	@echo " make octave - compile and test octave interfaces"
	@echo " make all - do all of the above"
	@echo " make spreadtest - compile and run spreader tests only"
	@echo " make clean - remove all object and executable files apart from MEX"
	@echo "For faster (multicore) making, append the flag -j"
	@echo ""
	@echo "Compile options: make [task] PREC=SINGLE for single-precision"
	@echo " make [task] OMP=OFF for single-threaded (otherwise openmp)"
	@echo " Don't forget to make clean before changing such options!"
	@echo ""
	@echo "To make python/python3 interfaces, see docs/install.rst"

# implicit rules for objects (note -o ensures writes to correct dir)
%.o: %.cpp %.h
	$(CXX) -c $(CXXFLAGS) $< -o $@
%.o: %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@
%.o: %.f %.h
	$(FC) -c $(FFLAGS) $< -o $@

# build the library...
lib: lib-static/libfinufft.a lib/libfinufft.so
	echo "lib-static/libfinufft.a and lib/libfinufft.so built"
lib-static/libfinufft.a: $(OBJS) $(HEADERS)
	ar rcs lib-static/libfinufft.a $(OBJS)
lib/libfinufft.so: $(OBJS) $(HEADERS)
	$(CXX) -shared $(OBJS) -o lib/libfinufft.so      # fails in mac osx
# see: http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html

# examples in C++ and C...                *** TO FIX C CAN'T FIND C++ HEADERS
EX = examples/example1d1$(SUFFIX)
EXC = examples/example1d1c$(SUFFIX)
examples: $(EX) $(EXC)
	$(EX)
	$(EXC)
$(EX): $(EX).o lib-static/libfinufft.a
	$(CXX) $(CXXFLAGS) $(EX).o lib-static/libfinufft.a $(LIBSFFT) -o $(EX)
$(EXC): $(EXC).o lib-static/libfinufft.a
	$(CC) $(CFLAGS) $(EXC).o lib-static/libfinufft.a $(LIBSFFT) $(CLINK) -o $(EXC)

# validation tests... (most link to .o allowing testing pieces separately)
test: lib-static/libfinufft.a test/testutils test/finufft1d_test test/finufft2d_test test/finufft3d_test test/dumbinputs
	(cd test; \
	export FINUFFT_REQ_TOL=$(REQ_TOL); \
	export FINUFFT_CHECK_TOL=$(CHECK_TOL); \
	./check_finufft.sh)
test/testutils: test/testutils.cpp src/utils.o src/utils.h $(HEADERS)
	$(CXX) $(CXXFLAGS) test/testutils.cpp src/utils.o -o test/testutils
test/finufft1d_test: test/finufft1d_test.cpp $(OBJS1) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft1d_test.cpp $(OBJS1) $(LIBSFFT) -o test/finufft1d_test
test/finufft2d_test: test/finufft2d_test.cpp $(OBJS2) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft2d_test.cpp $(OBJS2) $(LIBSFFT) -o test/finufft2d_test
test/finufft3d_test: test/finufft3d_test.cpp $(OBJS3) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft3d_test.cpp $(OBJS3) $(LIBSFFT) -o test/finufft3d_test
test/dumbinputs: test/dumbinputs.cpp lib-static/libfinufft.a $(HEADERS)
	$(CXX) $(CXXFLAGS) test/dumbinputs.cpp lib-static/libfinufft.a $(LIBSFFT) -o test/dumbinputs

# performance tests...
perftest: test/spreadtestnd test/finufft1d_test test/finufft2d_test test/finufft3d_test
# here the tee cmd copies output to screen. 2>&1 grabs both stdout and stderr...
	(cd test; ./spreadtestnd.sh 2>&1 | tee results/spreadtestnd_results.txt)
	(cd test; ./nuffttestnd.sh 2>&1 | tee results/nuffttestnd_results.txt)
test/spreadtestnd: test/spreadtestnd.cpp $(SOBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/spreadtestnd.cpp $(SOBJS) -o test/spreadtestnd

# spreader only test (useful for development work on spreader)...
spreadtest: test/spreadtestnd
	test/spreadtestnd 1 8e6 8e6 1e-6 1 0
	test/spreadtestnd 2 8e6 8e6 1e-6 1 0
	test/spreadtestnd 3 8e6 8e6 1e-6 1 0

# fortran interface...
F1=fortran/nufft1d_demo$(SUFFIX)
F2=fortran/nufft2d_demo$(SUFFIX)
F3=fortran/nufft3d_demo$(SUFFIX)
fortran: $(FOBJS) $(OBJS) $(HEADERS)
	$(FC) $(FFLAGS) $(F1).f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o $(F1)
	$(FC) $(FFLAGS) $(F2).f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o $(F2)
	$(FC) $(FFLAGS) $(F3).f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o $(F3)
	time -p $(F1)
	time -p $(F2)
	time -p $(F3)

# matlab .mex* executable...
matlab: lib-static/libfinufft.a $(HEADERS) matlab/finufft_m.cpp
ifeq ($(PREC),SINGLE)
	@echo "MATLAB interface only supports double precision; doing nothing"
else
	$(MEX) matlab/finufft.cpp lib-static/libfinufft.a matlab/finufft_m.cpp $(MFLAGS) $(LIBSFFT) -output matlab/finufft
endif

# octave .mex executable... (notice also creates matlab/finufft.o, but why?)
octave: lib-static/libfinufft.a $(HEADERS) matlab/finufft_m.cpp
ifeq ($(PREC),SINGLE)
	@echo "Octave interface only supports double precision; doing nothing"
else
	(cd matlab; mkoctfile --mex finufft.cpp ../lib-static/libfinufft.a finufft_m.cpp $(OFLAGS) $(LIBSFFT) -output finufft)
	@echo "Running octave interface test; please wait a few seconds..."
	(cd matlab; octave check_finufft.m)
endif

# for experts; force rebuilds fresh MEX (matlab/octave) gateway via mwrap...
# (needs mwrap)
mex: matlab/finufft.mw
	(cd matlab;\
	$(MWRAP) -list -mex finufft -cppcomplex -mb finufft.mw ;\
	$(MWRAP) -mex finufft -c finufft.cpp -cppcomplex finufft.mw )

# various obscure tests...
# This was for a CCQ application; zgemm was 10x faster!
manysmallprobs: lib-static/libfinufft.a $(HEADERS) test/manysmallprobs.cpp
	$(CXX) $(CXXFLAGS) test/manysmallprobs.cpp lib-static/libfinufft.a -o test/manysmallprobs $(LIBSFFT)
	(export OMP_NUM_THREADS=1; time test/manysmallprobs; unset OMP_NUM_THREADS)

# cleaning up...
clean:
	rm -f $(OBJS) $(SOBJS)
	rm -f lib/libfinufft.so lib-static/libfinufft.a
	rm -f test/spreadtestnd test/finufft?d_test test/testutils test/manysmallprobs test/results/*.out fortran/*.o fortran/nufft?d_demo fortran/nufft?d_demof examples/*.o examples/example1d1 examples/example1d1cexamples/example1d1f examples/example1d1cf matlab/*.o

# for experts; only do this if you have mwrap to rebuild the interfaces...
mexclean:
	rm -f matlab/finufft.cpp matlab/finufft?d?.m matlab/finufft.mex*
