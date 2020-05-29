# Makefile for FINUFFT.
# Barnett 2/12/19. Malleo's expansion for guru interface, summer 2019.
# Barnett tidying Feb, May 2020. Libin Lu edits, 2020.

# For simplicity, this is the only makefile; there are no makefiles in
# subdirectories. This makefile is useful to show humans how to compile
# FINUFFT and its various language interfaces and examples.
# Users should not need to edit this makefile (doing so would make it hard to
# stay up to date with the repo version). Rather, in order to change
# OS/environment-specific compilers and flags, create the file make.inc, which
# overrides the defaults below (which are for an ubuntu linux/GCC system).
# See docs/install.rst, and make.inc.* for examples.

# compilers, and linking from C, fortran...
CXX = g++
CC = gcc
FC = gfortran
CLINK = -lstdc++
FLINK = $(CLINK)
# compile flags for GCC, baseline single-threaded, double precision case...
# Notes: 1) -Ofast breaks isfinite() & isnan(), so use -O3 which now is as fast
#        2) -fcx-limited-range for fortran-speed complex arith in C++
CFLAGS = -fPIC -O3 -funroll-loops -march=native -fcx-limited-range
# tell examples where to find header files...
CFLAGS += -I include
FFLAGS = $(CFLAGS) -I fortran -I /usr/include
CXXFLAGS = $(CFLAGS) -std=c++14 -DNEED_EXTERN_C
# FFTW base name, and math linking...
FFTWNAME = fftw3
# the following uses fftw3_omp, since 10% faster than fftw3_threads...
FFTWOMPSUFFIX = omp
LIBS = -lm
# extra flags for multithreaded: C++/C/Fortran, MATLAB, and octave...
OMPFLAGS = -fopenmp
OMPLIBS = -lgomp
MOMPFLAGS = -lgomp -D_OPENMP
OOMPFLAGS = -lgomp
# flags for MATLAB MEX compilation...
MFLAGS = -largeArrayDims
# location of MATLAB's mex compiler...
MEX = mex
# flags for octave mkoctfile...
OFLAGS =
# For experts, location of MWrap executable (see docs/install.rst):
MWRAP = mwrap

# For your OS, override the above by placing make variables in make.inc ...
# (Please look in make.inc.* for ideas)
-include make.inc

# choose the precision (affects library names, test precisions)...
ifeq ($(PREC),SINGLE)
CXXFLAGS += -DSINGLE
CFLAGS += -DSINGLE
# note that PRECSUFFIX is used to choose fftw lib name, and also our demo names
PRECSUFFIX = f
REQ_TOL = 1e-6
CHECK_TOL = 2e-4
else
PRECSUFFIX =
REQ_TOL = 1e-12
CHECK_TOL = 1e-11
endif
# build (since fftw has many) names of libs to link against...
FFTW = $(FFTWNAME)$(PRECSUFFIX)
LIBSFFT = -l$(FFTW) $(LIBS)

# multi-threaded libs & flags (see defs above; note fftw3_threads slower)...
ifneq ($(OMP),OFF)
CXXFLAGS += $(OMPFLAGS)
CFLAGS += $(OMPFLAGS)
FFLAGS += $(OMPFLAGS)
MFLAGS += $(MOMPFLAGS)
OFLAGS += $(OOMPFLAGS)
LIBS += $(OMPLIBS)
LIBSFFT = -l$(FFTW) -l$(FFTW)_$(FFTWOMPSUFFIX) $(OMPLIBS)
endif

# decide name of obj files and finufft library we're building...
LIBNAME = libfinufft$(PRECSUFFIX)
DYNLIB = lib/$(LIBNAME).so
STATICLIB = lib-static/$(LIBNAME).a
# ======================================================================


# spreader object files
SOBJS = src/spreadinterp.o src/utils.o

# main library object files
OBJS = src/finufft.o src/simpleinterfaces.o src/common.o contrib/legendre_rule_fast.o $(SOBJS) fortran/finufft_f.o

.PHONY: usage lib examples test perftest fortran matlab octave all mex python clean objclean pyclean mexclean wheel docker-wheel

default: usage

all: test perftest lib examples fortran matlab octave python

usage:
	@echo "Makefile for FINUFFT library. Specify what to make:"
	@echo " make lib - compile the main library (in lib/ and lib-static/)"
	@echo " make examples - compile and run codes in examples/"
	@echo " make test - compile and run quick math validation tests"
	@echo " make perftest - compile and run performance tests"
	@echo " make fortran - compile and run Fortran examples"
	@echo " make matlab - compile MATLAB interfaces"
	@echo " make octave - compile and test octave interfaces"
	@echo " make python - compile and test python interfaces"	
	@echo " make all - do all the above (around 1 minute; assumes you have MATLAB, etc)"
	@echo " make spreadtest - compile and run spreader tests only"
	@echo " make objclean - remove all object files, preserving lib & MEX"
	@echo " make clean - also remove lib, MEX, py, and demo executables"
	@echo "For faster (multicore) making, append the flag -j"
	@echo ""
	@echo "Compile options: 'make [task] PREC=SINGLE' for single-precision"
	@echo " 'make [task] OMP=OFF' for single-threaded (otherwise OpenMP)"
	@echo " You must 'make objclean' before changing such options!"
	@echo ""
	@echo "Also see docs/install.rst"

# implicit rules for objects (note -o ensures writes to correct dir)
%.o: %.cpp %.h
	$(CXX) -c $(CXXFLAGS) $< -o $@
%.o: %.c %.h
	$(CC) -c $(CFLAGS) $< -o $@
%.o: %.f %.h
	$(FC) -c $(FFLAGS) $< -o $@

# included auto-generated code dependency...
src/spreadinterp.o: src/ker_horner_allw_loop.c src/ker_lowupsampfac_horner_allw_loop.c

# build the library...
lib: $(STATICLIB) $(DYNLIB)

ifeq ($(OMP),OFF)
	echo "$(STATICLIB) and $(DYNLIB) built, single-thread versions"
else
	echo "$(STATICLIB) and $(DYNLIB) built, multithreaded versions"
endif

$(STATICLIB): $(OBJS) 
	ar rcs $(STATICLIB) $(OBJS) 
$(DYNLIB): $(OBJS) 
	$(CXX) -shared $(OMPFLAGS) $(OBJS)  -o $(DYNLIB) $(LIBSFFT)

# here $(OMPFLAGS) and $(LIBSFFT) is needed for linking under mac osx.
# see: http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html
# Also note -l libs come after objects, as per modern GCC requirement.

# Examples in C++ and C... (exist separate codes for double vs single prec)
EX = examples/example1d1$(PRECSUFFIX)
EXC = examples/example1d1c$(PRECSUFFIX)
EX2 = examples/example2d1
EXG = examples/guru1d1
EXS = $(EX) $(EXC) $(EX2) $(EXG)

examples: $(EXS)
# use shell script to execute all in list. shell doesn't use $(E); $$ escapes $
	(for E in $(EXS); do ./$$E; done)

$(EX): $(EX).o $(DYNLIB)
	$(CXX) $(CXXFLAGS) $(EX).o $(DYNLIB) $(LIBSFFT) -o $(EX)
$(EX2): $(EX2).o $(DYNLIB)
	$(CXX) $(CXXFLAGS) $(EX2).o $(DYNLIB) $(LIBSFFT) -o $(EX2)
$(EXC): $(EXC).o $(DYNLIB)
	$(CC) $(CFLAGS) $(EXC).o $(DYNLIB) $(LIBSFFT) $(CLINK) -o $(EXC)
$(EXG): $(EXG).o $(DYNLIB)
	$(CXX) $(CXXFLAGS) $(EXG).o $(DYNLIB) $(LIBSFFT) -o $(EXG)

# validation tests... (some link to .o allowing testing pieces separately)
TESTS = test/testutils test/finufft1d_test test/finufft2d_test test/finufft3d_test test/dumbinputs test/finufft3dmany_test test/finufft2dmany_test test/finufft1dmany_test test/finufftGuru_test test/finufft1d_basicpassfail

# slow FTs in C++, for testing only
DO1 = test/directft/dirft1d.o
DO2 = test/directft/dirft2d.o
DO3 = test/directft/dirft3d.o

test: $(STATICLIB) $(TESTS)
	test/finufft1d_basicpassfail 
	(cd test; \
	export FINUFFT_REQ_TOL=$(REQ_TOL); \
	export FINUFFT_CHECK_TOL=$(CHECK_TOL); \
	./check_finufft.sh)

test/finufft1d_basicpassfail: test/finufft1d_basicpassfail.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) test/finufft1d_basicpassfail.cpp $(OBJS) $(LIBSFFT) -o test/finufft1d_basicpassfail
test/testutils: test/testutils.cpp src/utils.o
	$(CXX) $(CXXFLAGS) test/testutils.cpp src/utils.o $(LIBS) -o test/testutils
test/dumbinputs: test/dumbinputs.cpp $(DYNLIB) $(DO1)
	$(CXX) $(CXXFLAGS) test/dumbinputs.cpp $(OBJS) $(DO1) $(LIBSFFT) -o test/dumbinputs
test/finufft1d_test: test/finufft1d_test.cpp $(OBJS) $(DO1)
	$(CXX) $(CXXFLAGS) test/finufft1d_test.cpp $(OBJS) $(DO1) $(LIBSFFT) -o test/finufft1d_test
test/finufft2d_test: test/finufft2d_test.cpp $(OBJS) $(DO2)
	$(CXX) $(CXXFLAGS) test/finufft2d_test.cpp $(OBJS) $(DO2) $(LIBSFFT) -o test/finufft2d_test
test/finufft3d_test: test/finufft3d_test.cpp $(OBJS) $(DO3)
	$(CXX) $(CXXFLAGS) test/finufft3d_test.cpp $(OBJS) $(DO3) $(LIBSFFT) -o test/finufft3d_test
test/finufft1dmany_test: test/finufft1dmany_test.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) test/finufft1dmany_test.cpp $(OBJS) $(LIBSFFT) -o test/finufft1dmany_test
test/finufft2dmany_test: test/finufft2dmany_test.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) test/finufft2dmany_test.cpp $(OBJS) $(LIBSFFT) -o test/finufft2dmany_test
test/finufft3dmany_test: test/finufft3dmany_test.cpp $(OBJS) 
	$(CXX) $(CXXFLAGS) test/finufft3dmany_test.cpp $(OBJS) $(LIBSFFT) -o test/finufft3dmany_test
test/finufftGuru_test: test/finufftGuru_test.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) test/finufftGuru_test.cpp $(OBJS) $(LIBSFFT) -o test/finufftGuru_test


# performance tests...
perftest: test/spreadtestnd test/finufft1d_test test/finufft2d_test test/finufft3d_test
# here the tee cmd copies output to screen. 2>&1 grabs both stdout and stderr...
	(cd test; ./spreadtestnd.sh 2>&1 | tee results/spreadtestnd_results.txt)
	(cd test; ./nuffttestnd.sh 2>&1 | tee results/nuffttestnd_results.txt)
test/spreadtestnd: test/spreadtestnd.cpp $(SOBJS) 
	$(CXX) $(CXXFLAGS) test/spreadtestnd.cpp $(SOBJS) $(LIBS) -o test/spreadtestnd

# spreader only test (useful for development work on spreader)...
spreadtest: test/spreadtestnd
	test/spreadtestnd 1 8e6 8e6 1e-6 1 0
	test/spreadtestnd 2 8e6 8e6 1e-6 1 0
	test/spreadtestnd 3 8e6 8e6 1e-6 1 0

# --------------- LANGUAGE INTERFACES -----------------------

# fortran interface...
FT = fortran/test
FTOBJS = $(FT)/dirft1d.o $(FT)/dirft2d.o $(FT)/dirft3d.o $(FT)/dirft1df.o $(FT)/dirft2df.o $(FT)/dirft3df.o $(FT)/prini.o
FE = fortran/examples
F1 = $(FE)/example1d1$(PRECSUFFIX)
F2 = $(FE)/example1d1_guru$(PRECSUFFIX)
F3 = $(FE)/nufft1d_demo$(PRECSUFFIX)
F4 = $(FE)/nufft2d_demo$(PRECSUFFIX)
F5 = $(FE)/nufft3d_demo$(PRECSUFFIX)
F6 = $(FE)/nufft2dmany_demo$(PRECSUFFIX)
# GNU make trick to get list of executables to compile... (how auto 1 2... ?)
#F = $(foreach V, 1 2 3 4 5 6, $(F$V))
# too fancy; don't need.
# *** todo: make DYNLIB, but need to add to user's dyn lib path or exec only
# works from the top-level dir:
fortran: $(FTOBJS) $(STATICLIB)
	$(FC) $(FFLAGS) $(F1).f $(STATICLIB) $(LIBSFFT) $(FLINK) -o $(F1)
	for i in $(F1); do ./$$i; done
# (that was a bash script loop; note $$'s here are escaped dollar signs)

# matlab .mex* executable... (not worth starting matlab to test it)
matlab: $(STATICLIB)
ifeq ($(PREC),SINGLE)
	@echo "MATLAB interface only supports double precision; doing nothing"
else
	$(MEX) -DR2008OO matlab/nufft_plan_mex.cpp $(STATICLIB) -Iinclude $(MFLAGS) $(LIBSFFT) -output matlab/nufft_plan_mex
endif

# octave .mex executable... (also creates matlab/finufft.o for some reason)
octave: $(STATICLIB)
ifeq ($(PREC),SINGLE)
	@echo "Octave interface only supports double precision; doing nothing"
else
	(cd matlab; mkoctfile --mex -DR2008OO nufft_plan_mex.cpp -I../include ../$(STATICLIB) $(OFLAGS) $(LIBSFFT) -output nufft_plan_mex)
	@echo "Running octave interface test; please wait a few seconds..."
	(cd matlab; octave test/guru1dtest.m)
endif

# for experts: force rebuilds fresh MEX (matlab/octave) gateway via mwrap...
# (needs mwrap)
mex: matlab/nufft_plan.mw
	(cd matlab;\
	$(MWRAP) -mex nufft_plan_mex -c nufft_plan_mex.cpp -mb -cppcomplex nufft_plan.mw)

# python interfaces (v3 assumed)...
python: $(STATICLIB)
ifeq ($(PREC),SINGLE)
	@echo "python interface only supports double precision; doing nothing"
else
	(export FINUFFT_DIR=$(shell pwd); cd python; pip install .)
	python python/test/python_guru1d1.py	
	python python/test/demo1d1.py
	python python/test/run_accuracy_tests.py
endif

# *** please document these in make tasks echo above...:
wheel: $(STATICLIB)
	(export FINUFFT_DIR=$(shell pwd); cd python; python -m pip wheel . -w wheelhouse; delocate-wheel -w fixed_wheel -v wheelhouse/finufftpy*.whl)

docker-wheel:
	docker run --rm -e package_name=finufftpy -v `pwd`:/io quay.io/pypa/manylinux2010_x86_64 /io/python/ci/build-wheels.sh



# ------------- Various obscure/devel tests -----------------
# This was for a CCQ application; zgemm was 10x faster!
test/manysmallprobs: $(STATICLIB)  test/manysmallprobs.cpp
	$(CXX) $(CXXFLAGS) test/manysmallprobs.cpp $(STATICLIB) -o test/manysmallprobs $(LIBSFFT)
#	@echo "manysmallprobs: all avail threads..."
#	test/manysmallprobs	
	@echo "manysmallprobs: single-thread..."
	OMP_NUM_THREADS=1 test/manysmallprobs


# ------------- Cleaning up (including all versions of lib, and interfaces)...
clean: objclean pyclean
	rm -f lib-static/*.a lib/*.so
	rm -f matlab/*.mex*
	rm -f $(TESTS) test/results/*.out fortran/*_demo fortran/*_demof $(EXS) examples/example1d1 examples/example1d1c examples/example1d1f examples/example1d1cf

# this is needed before changing precision or threading...
objclean:
	rm -f $(OBJS) test/directft/*.o test/*.o
	rm -f fortran/*.o examples/*.o matlab/*.o

pyclean:
	rm -f python/finufftpy/*.pyc python/finufftpy/__pycache__/* python/test/*.pyc python/test/__pycache__/*
	rm -rf python/fixed_wheel python/wheelhouse

# for experts; only do this if you have mwrap to rebuild the interfaces!
mexclean:
	rm -f matlab/finufft.cpp matlab/finufft?d?.m matlab/finufft?d?many.m matlab/finufft.mex*
