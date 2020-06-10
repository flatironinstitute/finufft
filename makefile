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

# compilers, and linking from C, fortran. We use GCC by default...
CXX = g++
CC = gcc
FC = gfortran
CLINK = -lstdc++
FLINK = $(CLINK)
# compile flags for GCC, baseline single-threaded, double precision case...
# Notes: 1) -Ofast breaks isfinite() & isnan(), so use -O3 which now is as fast
#        2) -fcx-limited-range for fortran-speed complex arith in C++.
#        3) we use simply-expanded makefile variables, otherwise confusing.
CFLAGS := -O3 -funroll-loops -march=native -fcx-limited-range
FFLAGS := $(CFLAGS)
CXXFLAGS := $(CFLAGS)
# FFTW base name, and math linking...
FFTWNAME = fftw3
# the following uses fftw3_omp, since 10% faster than fftw3_threads...
FFTWOMPSUFFIX = omp
LIBS := -lm
# multithreading for GCC: C++/C/Fortran, MATLAB, and octave (ICC differs)...
OMPFLAGS = -fopenmp
OMPLIBS = -lgomp
MOMPFLAGS = -lgomp -D_OPENMP
OOMPFLAGS = -lgomp
# MATLAB MEX compilation (OO for new interface; int64 for mwrap 0.33.9)...
MFLAGS := -largeArrayDims -DR2008OO -D_INT64_T
# location of MATLAB's mex compiler (could add flags to switch GCC, etc)...
MEX = mex
# octave mkoctfile...
OFLAGS = -DR2008OO -D_INT64_T
# For experts only, location of MWrap executable (see docs/install.rst):
MWRAP = mwrap
# absolute path of this makefile, ie FINUFFT's top-level directory...
FINUFFT = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# For your OS, override the above by placing make variables in make.inc ...
# (Please look in make.inc.* for ideas)
-include make.inc

# now come flags that should be added, whatever you overrode in make.inc
# to prevent 
# tell tests & examples where to find header files...
INCL = -Iinclude
# the NEED_EXTERN_C directive tells common.cpp to include plain C header
# -fPIC (position-indep code) needed to build dyn lib (.so)
# Also, we force return (via :=) to the land of simply-expanded variables...
CXXFLAGS := $(CXXFLAGS) $(INCL) -fPIC -std=c++14 -DNEED_EXTERN_C
CFLAGS := $(CFLAGS) $(INCL) -fPIC
# /usr/include needed for fftw3.f...
FFLAGS := $(FFLAGS) $(INCL) -I/usr/include -fPIC

# choose the precision (affects library names, test precisions)...
ifeq ($(PREC),SINGLE)
CXXFLAGS += -DSINGLE
CFLAGS += -DSINGLE
# note that PRECSUFFIX appends the fftw lib names, and also our demo names
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
LIBSFFT = -l$(FFTW) -l$(FFTW)_$(FFTWOMPSUFFIX) $(LIBS)
endif

# decide name of obj files and finufft library we're building...
LIBNAME = libfinufft$(PRECSUFFIX)
DYNLIB = lib/$(LIBNAME).so
STATICLIB = lib-static/$(LIBNAME).a
# absolute path to the .so, useful for portable executables...
ABSDYNLIB = $(FINUFFT)/$(DYNLIB)
# ======================================================================


# spreader object files
SOBJS = src/spreadinterp.o src/utils.o

# main library object files
OBJS = src/finufft.o src/simpleinterfaces.o src/common.o contrib/legendre_rule_fast.o $(SOBJS) fortran/finufft_f.o fortran/finufft_f_legacy.o julia/finufft_j.o

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
	@echo " make objclean - remove all object files, preserving libs & MEX"
	@echo " make clean - also remove all lib, MEX, py, and demo executables"
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
	$(CXX) -shared $(OMPFLAGS) $(OBJS) -o $(DYNLIB) $(LIBSFFT)

# here $(OMPFLAGS) and $(LIBSFFT) is needed for linking under mac osx.
# see: http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html
# Also note -l libs come after objects, as per modern GCC requirement.

# Examples in C++ and C... (single prec codes separate, and not all have one)
EX = examples/example1d1$(PRECSUFFIX)
EXC = examples/example1d1c$(PRECSUFFIX)
EX2 = examples/example2d1
EXG = examples/guru1d1
ifeq ($(PREC),SINGLE)
EXS = $(EX) $(EXC)
else
EXS = $(EX) $(EXC) $(EX2) $(EXG)
endif

examples: $(EXS)
# use shell script to execute all in list. shell doesn't use $(E); $$ escapes $
	(for E in $(EXS); do ./$$E; done)

# compile examples; note absolute .so path so executable anywhere, dep libs not needed to be listed...
$(EX): $(EX).o $(DYNLIB)
	$(CXX) $(CXXFLAGS) $(EX).o $(ABSDYNLIB) -o $(EX)
$(EXC): $(EXC).o $(DYNLIB)
	$(CC) $(CFLAGS) $(EXC).o $(ABSDYNLIB) $(LIBSFFT) $(CLINK) -o $(EXC)
$(EX2): $(EX2).o $(DYNLIB)
	$(CXX) $(CXXFLAGS) $(EX2).o $(ABSDYNLIB) -o $(EX2)
$(EXG): $(EXG).o $(DYNLIB)
	$(CXX) $(CXXFLAGS) $(EXG).o $(ABSDYNLIB) -o $(EXG)

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

# these all link to .o rather than the lib.so, for simplicity...
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
FD = fortran/directft
# CMCL NUFFT fortran test codes (only needed by the nufft*_demo* codes)
CMCLOBJS = $(FD)/dirft1d.o $(FD)/dirft2d.o $(FD)/dirft3d.o $(FD)/dirft1df.o $(FD)/dirft2df.o $(FD)/dirft3df.o $(FD)/prini.o
FE = fortran/examples
F1 = $(FE)/simple1d1$(PRECSUFFIX)
F2 = $(FE)/guru1d1$(PRECSUFFIX)
F3 = $(FE)/nufft1d_demo$(PRECSUFFIX)
F4 = $(FE)/nufft1d_demo_legacy$(PRECSUFFIX)
F5 = $(FE)/nufft2d_demo$(PRECSUFFIX)
F6 = $(FE)/nufft3d_demo$(PRECSUFFIX)
F7 = $(FE)/nufft2dmany_demo$(PRECSUFFIX)
# GNU make trick to get list of executables to compile... (how auto 1 2... ?)
F = $(foreach V, 1 2 3 4 5 6 7, $(F$V))
fortran: $(CMCLOBJS) $(DYNLIB)
	for i in $(F); do \
	$(FC) $(FFLAGS) $$i.f $(CMCLOBJS) $(ABSDYNLIB) $(FLINK) -o $$i ; \
	./$$i ; \
	done
# (that was a bash script loop; note $$'s here are escaped dollar signs)

# matlab .mex* executable... (not worth starting matlab to test it)
# note various -D defines; INT64_T needed for mwrap 0.33.9.
matlab: $(STATICLIB)
ifeq ($(PREC),SINGLE)
	@echo "MATLAB interface only supports double precision; doing nothing"
else
	$(MEX) matlab/finufft_plan_mex.cpp $(STATICLIB) -Iinclude $(MFLAGS) $(LIBSFFT) -output matlab/finufft_plan_mex
endif

# octave .mex executable... (also creates matlab/finufft.o for some reason)
octave: $(STATICLIB)
ifeq ($(PREC),SINGLE)
	@echo "Octave interface only supports double precision; doing nothing"
else
	(cd matlab; mkoctfile --mex finufft_plan_mex.cpp -I../include ../$(STATICLIB) $(OFLAGS) $(LIBSFFT) -output finufft_plan_mex)
	@echo "Running octave interface test; please wait a few seconds..."
	(cd matlab; octave test/guru1dtest.m)
endif

# for experts: force rebuilds fresh MEX (matlab/octave) gateway via mwrap...
# (needs mwrap, moreover the correct version, eg 0.33.9)
mex: matlab/finufft_plan.mw
	(cd matlab;\
	$(MWRAP) -mex finufft_plan_mex -c finufft_plan_mex.cpp -mb -cppcomplex finufft_plan.mw)

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

# python packaging: *** please document these in make tasks echo above...
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


# ------------- Cleaning up (including *all* versions of lib, and interfaces)...
clean: objclean pyclean
	rm -f $(STATICLIB) $(DYNLIB)
	rm -f matlab/*.mex*
	rm -f $(TESTS) test/results/*.out
# recursion hack to clean up all example single-prec, executables too...
# (note it's possible to clean single only, but not double only)
ifneq ($(PREC),SINGLE)
	make clean PREC=SINGLE
else
	rm -f $(EXS) $(F)
endif

# indiscriminate .o killer: needed before changing precision or threading...
objclean:
	rm -f $(OBJS) test/directft/*.o test/*.o
	rm -f fortran/*.o fortran/examples/*.o examples/*.o matlab/*.o $(CMCLOBJS)

pyclean:
	rm -f python/finufftpy/*.pyc python/finufftpy/__pycache__/* python/test/*.pyc python/test/__pycache__/*
	rm -rf python/fixed_wheel python/wheelhouse

# for experts; only do this if you have mwrap to rebuild the interfaces!
mexclean:
	rm -f matlab/finufft.cpp matlab/finufft?d?.m matlab/finufft?d?many.m matlab/finufft.mex*
