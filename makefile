# Makefile for FINUFFT, v1.2

# For simplicity, this is the only makefile; there are no makefiles in
# subdirectories. This makefile is useful to show humans how to compile
# FINUFFT and its various language interfaces and examples.
# Users should not need to edit this makefile (doing so would make it hard to
# stay up to date with the repo version). Rather, in order to change
# OS/environment-specific compilers and flags, create the file make.inc, which
# overrides the defaults below (which are for an ubuntu linux/GCC system).
# See docs/install.rst, and make.inc.* for examples.

# Barnett 2017-2020. Malleo's expansion for guru interface, summer 2019.
# Barnett tidying Feb, May 2020. Libin Lu edits, 2020.
# Garrett Wright dual-prec lib build, June 2020.

# compilers, and linking from C, fortran. We use GCC by default...
CXX = g++
CC = gcc
FC = gfortran
CLINK = -lstdc++
FLINK = $(CLINK)
# baseline compile flags for GCC (no multithreading):
# Notes: 1) -Ofast breaks isfinite() & isnan(), so use -O3 which now is as fast
#        2) -fcx-limited-range for fortran-speed complex arith in C++.
#        3) we use simply-expanded makefile variables, otherwise confusing.
CFLAGS := -O3 -funroll-loops -march=native -fcx-limited-range
FFLAGS := $(CFLAGS)
CXXFLAGS := $(CFLAGS)
# FFTW base name, and math linking...
FFTWNAME = fftw3
# linux default is fftw3_omp, since 10% faster than fftw3_threads...
FFTWOMPSUFFIX = omp
LIBS := -lm
# multithreading for GCC: C++/C/Fortran, MATLAB, and octave (ICC differs)...
OMPFLAGS = -fopenmp
OMPLIBS = -lgomp
MOMPFLAGS = -D_OPENMP
OOMPFLAGS =
# MATLAB MEX compilation (OO for new interface; int64 for mwrap 0.33.9)...
MFLAGS := -largeArrayDims -DR2008OO
# location of MATLAB's mex compiler (could add flags to switch GCC, etc)...
MEX = mex
# octave mkoctfile...
OFLAGS = -DR2008OO
# For experts only, location of MWrap executable (see docs/install.rst):
MWRAP = mwrap
# absolute path of this makefile, ie FINUFFT's top-level directory...
FINUFFT = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# For your OS, override the above by setting make variables in make.inc ...
# (Please look in make.inc.* for ideas)
-include make.inc

# Now come flags that should be added, whatever user overrode in make.inc.
# -fPIC (position-indep code) needed to build dyn lib (.so)
# Also, we force return (via :=) to the land of simply-expanded variables...
INCL = -Iinclude
CXXFLAGS := $(CXXFLAGS) $(INCL) -fPIC -std=c++14
CFLAGS := $(CFLAGS) $(INCL) -fPIC
# here /usr/include needed for fftw3.f "fortran header"...
FFLAGS := $(FFLAGS) $(INCL) -I/usr/include -fPIC

# double precision tolerances and test errors (`test' target)...
REQ_TOL = 1e-12
CHECK_TOL = 1e-11
# same for single-prec tests...
REQ_TOL_SINGLE = 1e-6
CHECK_TOL_SINGLE = 2e-4

# single-thread total list of math and FFTW libs (now both precisions)...
# (Note: finufft tests use LIBSFFT; lower-level tests only need LIBS)
LIBSFFT := -l$(FFTWNAME) -l$(FFTWNAME)f $(LIBS)

# multi-threaded libs & flags
ifneq ($(OMP),OFF)
CXXFLAGS += $(OMPFLAGS)
CFLAGS += $(OMPFLAGS)
FFLAGS += $(OMPFLAGS)
MFLAGS += $(MOMPFLAGS)
OFLAGS += $(OOMPFLAGS)
LIBS += $(OMPLIBS)
ifneq ($(MINGW),ON)
# omp override for total list of math and FFTW libs (now both precisions)...
LIBSFFT := -l$(FFTWNAME) -l$(FFTWNAME)_$(FFTWOMPSUFFIX) -l$(FFTWNAME)f -l$(FFTWNAME)f_$(FFTWOMPSUFFIX) $(LIBS)
endif
endif

# name & location of library we're building...
LIBNAME = libfinufft
DYNLIB = lib/$(LIBNAME).so
STATICLIB = lib-static/$(LIBNAME).a
# absolute path to the .so, useful for linking so executables portable...
ABSDYNLIB = $(FINUFFT)/$(DYNLIB)
# ======================================================================

# spreader is subset of the library with self-contained testing, hence own objs:
# double-prec spreader object files that also need single precision...
SOBJS = src/spreadinterp.o src/utils.o
# their single-prec versions
SOBJSF = $(SOBJS:%.o=%_32.o)
# precision-dependent spreader object files (compiled & linked only once)...
SOBJS_PI = src/utils_precindep.o
# spreader dual-precision objs
SOBJSD = $(SOBJS) $(SOBJSF) $(SOBJS_PI)

# double-prec library object files that also need single precision...
OBJS = $(SOBJS) src/finufft.o src/simpleinterfaces.o fortran/finufftfort.o
# their single-prec versions
OBJSF = $(OBJS:%.o=%_32.o)
# precision-dependent library object files (compiled & linked only once)...
OBJS_PI = $(SOBJS_PI) contrib/legendre_rule_fast.o julia/finufftjulia.o
# all lib dual-precision objs
OBJSD = $(OBJS) $(OBJSF) $(OBJS_PI)

.PHONY: usage lib examples test perftest fortran matlab octave all mex python clean objclean pyclean mexclean wheel docker-wheel

default: usage

all: test perftest lib examples fortran matlab octave python

usage:
	@echo "Makefile for FINUFFT library. Specify what to make:"
	@echo " make lib - compile the main library (in lib/ and lib-static/)"
	@echo " make examples - compile and run codes in examples/"
	@echo " make test - compile and run quick math validation tests (double only right now)"
	@echo " make perftest - compile and run performance tests"
	@echo " make fortran - compile and run Fortran tests and examples"
	@echo " make matlab - compile MATLAB interfaces"
	@echo " make octave - compile and test octave interfaces"
	@echo " make python - compile and test python interfaces"
	@echo " make all - do all the above (around 1 minute; assumes you have MATLAB, etc)"
	@echo " make spreadtest - compile & run spreader tests only (no FFTW)"
	@echo " make objclean - remove all object files, preserving libs & MEX"
	@echo " make clean - also remove all lib, MEX, py, and demo executables"
	@echo "For faster (multicore) making, append, for example, -j8"
	@echo ""
	@echo "Compile options:"
	@echo " 'make [task] OMP=OFF' for single-threaded (otherwise OpenMP)"
	@echo " You must 'make objclean' before changing such options!"
	@echo ""
	@echo "Also see docs/install.rst"

# collect headers for implicit depends
HEADERS = $(wildcard include/*.h)

# implicit rules for objects (note -o ensures writes to correct dir)
%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $< -o $@
%_32.o: %.cpp $(HEADERS)
	$(CXX) -DSINGLE -c $(CXXFLAGS) $< -o $@
%.o: %.c $(HEADERS)
	$(CC) -c $(CFLAGS) $< -o $@
%_32.o: %.c $(HEADERS)
	$(CC) -DSINGLE -c $(CFLAGS) $< -o $@
%.o: %.f $(HEADERS)
	$(FC) -c $(FFLAGS) $< -o $@
%_32.o: %.f $(HEADERS)
	$(FC) -DSINGLE -c $(FFLAGS) $< -o $@

# included auto-generated code dependency...
src/spreadinterp.o: src/ker_horner_allw_loop.c src/ker_lowupsampfac_horner_allw_loop.c

# spreader only test, double/single (useful for development work on spreader)...
spreadtest: test/spreadtestnd test/spreadtestnd_32
	@echo "running double then single precision spreader tests..."
	test/spreadtestnd 1 8e6 8e6 1e-6
	test/spreadtestnd 2 8e6 8e6 1e-6
	test/spreadtestnd 3 8e6 8e6 1e-6
	test/spreadtestnd_32 1 8e6 8e6 1e-3
	test/spreadtestnd_32 2 8e6 8e6 1e-3
	test/spreadtestnd_32 3 8e6 8e6 1e-3

# build library with double/single prec both bundled in...
lib: $(STATICLIB) $(DYNLIB)
$(STATICLIB): $(OBJSD)
ifeq ($(OMP),OFF)
	@echo "$(STATICLIB) built, single-thread version"
else
	@echo "$(STATICLIB) built, multithreaded version"
endif
	ar rcs $(STATICLIB) $(OBJSD)
$(DYNLIB): $(OBJSD)
ifeq ($(OMP),OFF)
	@echo "$(DYNLIB) built, single-thread version"
else
	@echo "$(DYNLIB) built, multithreaded version"
endif
	$(CXX) -shared $(OMPFLAGS) $(OBJSD) -o $(DYNLIB) $(LIBSFFT)

# here $(OMPFLAGS) and $(LIBSFFT) is even needed for linking under mac osx.
# see: http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html
# Also note -l libs come after objects, as per modern GCC requirement.

# Examples in C++ and C... (single prec codes separate, and not all have one)
EXAMPLES = $(basename $(wildcard examples/*.*))
examples: $(EXAMPLES)
	@echo "Made: $(EXAMPLES)"

examples/%: examples/%.o $(DYNLIB)
	$(CXX) $(CXXFLAGS) $< $(ABSDYNLIB) -o $@
	./$@
examples/%c: examples/%c.o $(DYNLIB)
	$(CC) $(CFLAGS) $< $(ABSDYNLIB) $(LIBSFFT) $(CLINK) -o $@
	./$@
examples/%cf: examples/%cf.o $(DYNLIB)
	$(CC) $(CFLAGS) $< $(ABSDYNLIB) $(LIBSFFT) $(CLINK) -o $@
	./$@


# validation tests... (some link to .o allowing testing pieces separately)
TESTS = test/testutils test/finufft1d_test test/finufft2d_test test/finufft3d_test test/dumbinputs test/finufft3dmany_test test/finufft2dmany_test test/finufft1dmany_test test/finufftGuru_test test/finufft1d_basicpassfail

test: $(STATICLIB) $(TESTS)
	test/finufft1d_basicpassfail
	(cd test; \
	export FINUFFT_REQ_TOL=$(REQ_TOL); \
	export FINUFFTF_REQ_TOL=$(REQ_TOL_SINGLE); \
	export FINUFFT_CHECK_TOL=$(CHECK_TOL); \
	export FINUFFTF_CHECK_TOL=$(CHECK_TOL_SINGLE); \
	./check_finufft.sh)

# these all link to .o rather than the lib.so, allowing partial build tests...
# *** should they link lib.so?
# *** automate this make task better:
test/finufft1d_basicpassfail: test/finufft1d_basicpassfail.cpp $(OBJSD)
	$(CXX) $(CXXFLAGS) test/finufft1d_basicpassfail.cpp $(OBJSD) $(LIBSFFT) -o test/finufft1d_basicpassfail
test/testutils: test/testutils.cpp src/utils_precindep.o
	$(CXX) $(CXXFLAGS) test/testutils.cpp src/utils_precindep.o $(LIBS) -o test/testutils
test/dumbinputs: test/dumbinputs.cpp $(DYNLIB)
	$(CXX) $(CXXFLAGS) test/dumbinputs.cpp $(OBJSD) $(LIBSFFT) -o test/dumbinputs
test/finufft1d_test: test/finufft1d_test.cpp $(OBJSD)
	$(CXX) $(CXXFLAGS) test/finufft1d_test.cpp $(OBJSD) $(LIBSFFT) -o test/finufft1d_test
test/finufft2d_test: test/finufft2d_test.cpp $(OBJSD)
	$(CXX) $(CXXFLAGS) test/finufft2d_test.cpp $(OBJSD) $(LIBSFFT) -o test/finufft2d_test
test/finufft3d_test: test/finufft3d_test.cpp $(OBJSD)
	$(CXX) $(CXXFLAGS) test/finufft3d_test.cpp $(OBJSD) $(LIBSFFT) -o test/finufft3d_test
test/finufft1dmany_test: test/finufft1dmany_test.cpp $(OBJSD)
	$(CXX) $(CXXFLAGS) test/finufft1dmany_test.cpp $(OBJSD) $(LIBSFFT) -o test/finufft1dmany_test
test/finufft2dmany_test: test/finufft2dmany_test.cpp $(OBJSD)
	$(CXX) $(CXXFLAGS) test/finufft2dmany_test.cpp $(OBJSD) $(LIBSFFT) -o test/finufft2dmany_test
test/finufft3dmany_test: test/finufft3dmany_test.cpp $(OBJSD)
	$(CXX) $(CXXFLAGS) test/finufft3dmany_test.cpp $(OBJSD) $(LIBSFFT) -o test/finufft3dmany_test
test/finufftGuru_test: test/finufftGuru_test.cpp $(OBJSD)
	$(CXX) $(CXXFLAGS) test/finufftGuru_test.cpp $(OBJSD) $(LIBSFFT) -o test/finufftGuru_test

# performance tests...
perftest: test/spreadtestnd test/finufft1d_test test/finufft2d_test test/finufft3d_test
# here the tee cmd copies output to screen. 2>&1 grabs both stdout and stderr...
	(cd test; ./spreadtestnd.sh 2>&1 | tee results/spreadtestnd_results.txt)
	(cd test; ./nuffttestnd.sh 2>&1 | tee results/nuffttestnd_results.txt)
test/spreadtestnd: test/spreadtestnd.cpp $(SOBJSD)
	$(CXX) $(CXXFLAGS) test/spreadtestnd.cpp $(SOBJSD) $(LIBS) -o test/spreadtestnd
	$(CXX) $(CXXFLAGS) -DSINGLE test/spreadtestnd.cpp $(SOBJSD) $(LIBS) -o test/spreadtestnd_32

# --------------- LANGUAGE INTERFACES -----------------------

# fortran interface...
FD = fortran/directft
# CMCL NUFFT fortran test codes (only needed by the nufft*_demo* codes)
CMCLOBJS = $(FD)/dirft1d.o $(FD)/dirft2d.o $(FD)/dirft3d.o $(FD)/dirft1df.o $(FD)/dirft2df.o $(FD)/dirft3df.o $(FD)/prini.o
FE_DIR = fortran/examples
FE64 = $(FE_DIR)/simple1d1 $(FE_DIR)/guru1d1 $(FE_DIR)/nufft1d_demo $(FE_DIR)/nufft2d_demo $(FE_DIR)/nufft3d_demo $(FE_DIR)/nufft2dmany_demo
FE32 = $(FE64:%=%f)

#fortran target pattern match
$(FE_DIR)/%: $(FE_DIR)/%.f $(CMCLOBJS) $(DYNLIB)
	$(FC) $(FFLAGS) $< $(CMCLOBJS) $(ABSDYNLIB) $(FLINK) -o $@
	./$@
$(FE_DIR)/%f: $(FE_DIR)/%f.f $(CMCLOBJS) $(DYNLIB)
	$(FC) $(FFLAGS) $< $(CMCLOBJS) $(ABSDYNLIB) $(FLINK) -o $@
	./$@

fortran: $(FE64) $(FE32) $(CMCLOBJS) $(DYNLIB)


# matlab .mex* executable... (not worth starting matlab to test it)
# note various -D defines; INT64_T needed for mwrap 0.33.9.
matlab: $(STATICLIB)
	$(MEX) matlab/finufft.cpp $(STATICLIB) $(INCL) $(MFLAGS) $(LIBSFFT) -output matlab/finufft

# octave .mex executable... (also creates matlab/finufft.o for some reason)
octave: $(STATICLIB)
	(cd matlab; mkoctfile --mex finufft.cpp -I../include ../$(STATICLIB) $(OFLAGS) $(LIBSFFT) -output finufft)
	@echo "Running octave interface test; please wait a few seconds..."
	(cd matlab; octave test/guru1dtest.m)

# for experts: force rebuilds fresh MEX (matlab/octave) gateway via mwrap...
# (needs mwrap, moreover a recent version, eg 0.33.10)
mex: matlab/finufft.mw
	(cd matlab;\
	$(MWRAP) -mex finufft -c finufft.cpp -mb -cppcomplex finufft.mw)

# python interfaces (v3 assumed)...
python: $(STATICLIB) $(DYNLIB)
	(export FINUFFT_DIR=$(shell pwd); cd python; pip install .)
	python python/test/python_guru1d1.py
	python python/test/demo1d1.py
	python python/test/run_accuracy_tests.py

# python packaging: *** please document these in make tasks echo above...
wheel: $(STATICLIB) $(DYNLIB)
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
	rm -f $(EXAMPLES) $(FE64) $(FE32)

# indiscriminate .o killer (including old ones); needed before changing threading...
objclean:
	rm -f src/*.o test/directft/*.o test/*.o examples/*.o matlab/*.o
	rm -f fortran/*.o $(FE_DIR)/*.o $(FD)/*.o

# *** need to update this:
pyclean:
	rm -f python/finufftpy/*.pyc python/finufftpy/__pycache__/* python/test/*.pyc python/test/__pycache__/*
	rm -rf python/fixed_wheel python/wheelhouse

# for experts; only run this if you have mwrap to rebuild the interfaces!
mexclean:
	rm -f matlab/finufft_plan.m matlab/finufft.cpp matlab/finufft.mex*
