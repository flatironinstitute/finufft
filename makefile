# Makefile for FINUFFT.
# Barnett 2/12/19.

# This is the only makefile; there are no makefiles in subdirectories.
# Users should not need to edit this makefile (doing so would make it hard to
# stay up to date with the repo version). Rather, in order to change
# OS/environment-specific compilers and flags, create the file make.inc, which
# overrides the defaults below (which are for ubuntu linux/GCC system).
# See docs/install.rst, and make.inc.*

# compilers, and linking from C, fortran...
CXX=g++
CC=gcc
FC=gfortran
CLINK=-lstdc++
FLINK=$(CLINK)
# compile flags for GCC, baseline single-threaded, double precision case...
# Notes: 1) -Ofast breaks isfinite() & isnan(), so use -O3 which now is as fast
#        2) -fcx-limited-range for fortran-speed complex arith in C++
CFLAGS   =  -Wall -fPIC -O3 -funroll-loops -march=native -fcx-limited-range
# tell examples where to find header files...
CFLAGS   += -I include
FFLAGS   = $(CFLAGS)
CXXFLAGS = $(CFLAGS) -DNEED_EXTERN_C
# FFTW base name, and math linking...
FFTWNAME=fftw3
# the following uses fftw3_omp, since 10% faster than fftw3_threads...
FFTWOMPSUFFIX=omp
LIBS = -lm
# extra flags for multithreaded: C++/C/Fortran, MATLAB, and octave...
OMPFLAGS = -fopenmp
OMPLIBS =
MOMPFLAGS = -lgomp -D_OPENMP
OOMPFLAGS = -lgomp
# flags for MATLAB MEX compilation...
MFLAGS=-largeArrayDims
# location of MATLAB's mex compiler...
MEX=mex
# flags for octave mkoctfile...
OFLAGS=
# For experts, location of MWrap executable (see docs/install.rst):
MWRAP=mwrap

# For your OS, override the above by placing make variables in make.inc ...
# (Please look in make.inc.* for ideas)
-include make.inc


 # choose the precision (affects library names, test precisions)... 
ifeq ($(PREC),SINGLE) 
CXXFLAGS += -DSINGLE 
CFLAGS += -DSINGLE 
# note that PRECSUFFIX is used to choose fftw lib name, and also our demo names 
PRECSUFFIX=f
REQ_TOL = 1e-6 
CHECK_TOL = 2e-4 
else 
PRECSUFFIX= 
REQ_TOL = 1e-12 
CHECK_TOL = 1e-11 
endif


# build (since fftw has many) names of libs to link against...
#FFTW = # $(FFTWNAME) $(PRECSUFFIX)
LIBSFFT = -lfftw3f -lfftw3 $(LIBS)

# multi-threaded libs & flags (see defs above; note fftw3_threads slower)...
ifneq ($(OMP),OFF)
CXXFLAGS += $(OMPFLAGS)
CFLAGS += $(OMPFLAGS)
FFLAGS += $(OMPFLAGS)
MFLAGS += $(MOMPFLAGS)
OFLAGS += $(OOMPFLAGS)
LIBS += $(OMPLIBS)
LIBSFFT += -lfftw3_$(FFTWOMPSUFFIX) -lfftw3f_$(FFTWOMPSUFFIX) $(OMPLIBS)
endif

# decide name of obj files and finufft library we're building...
LIBNAME=libfinufft$(PRECSUFFIX)
DYNAMICLIB = lib/$(LIBNAME).so
STATICLIB = lib-static/$(LIBNAME).a
LEGLIB = lib-static/$(LIBNAME)_legacy.a
OLDLIB = lib-static/$(LIBNAME)_old.a
# ======================================================================

# objects to compile: spreader...
SOBJS = src/spreadinterp_tempinstant.o src/utils_tempinstant.o

#common objects
COBJS = src/common_tempinstant.o contrib/legendre_rule_fast.o

# LEGACY just the dimensions (1,2,3) separately...
LEG_OBJS1 =  src/legacy/finufft1d.o src/legacy/invokeGuru.o  
LEG_OBJS2 =  src/legacy/finufft2d.o src/legacy/invokeGuru.o 
LEG_OBJS3 =  src/legacy/finufft3d.o src/legacy/invokeGuru.o 

LEG_OBJS = src/legacy/finufft1d.o src/legacy/finufft2d.o src/legacy/finufft3d.o src/legacy/invokeGuru.o 

#OLD
OLD_OBJS1 = src/old/finufft1d_old.o src/direct/dirft1d.o
OLD_OBJS2 = src/old/finufft2d_old.o src/direct/dirft2d.o
OLD_OBJS3 = src/old/finufft3d_old.o src/direct/dirft3d.o

OLD_OBJS = $(OLD_OBJS1) $(OLD_OBJS2) $(OLD_OBJS3)

OBJS = src/finufft_tempinstant.o $(COBJS) $(SOBJS)

# for Fortran interface demos...
FOBJS = fortran/dirft1d.o fortran/dirft2d.o fortran/dirft3d.o fortran/dirft1df.o fortran/dirft2df.o fortran/dirft3df.o fortran/prini.o

HEADERS = include/spreadinterp_tempinstant.h include/finufft_tempinstant.h include/dirft.h include/common_tempinstant.h include/defs.h include/utils_tempinstant.h include/finufft_f.h

#LEG_HEADERS =
OLD_HEADERS = include/finufft_old.h

.PHONY: usage lib examples test perftest fortran matlab octave all mex python python3 clean objclean pyclean mexclean

default: usage

all: test perftest lib examples fortran matlab octave python3

usage:
	@echo "Makefile for FINUFFT library. Specify what to make:"
	@echo " make lib - compile the main library (in lib/ and lib-static/)"
	@echo " make examples - compile and run codes in examples/"
	@echo " make test - compile and run quick math validation tests"
	@echo " make perftest - compile and run performance tests"
	@echo " make fortran - compile and test Fortran interfaces"
	@echo " make matlab - compile MATLAB interfaces"
	@echo " make octave - compile and test octave interfaces"
	@echo " make python3 - compile and test python3 interfaces"	
	@echo " make all - do all the above (around 1 minute; assumes you have MATLAB, etc)"
	@echo " make python - compile and test python (v2) interfaces"
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
src/spreadinterp_tempinstant.o: src/ker_horner_allw_loop.c src/ker_lowupsampfac_horner_allw_loop.c

# build the library...
lib: $(STATICLIB) $(DYNAMICLIB)  $(LEGLIB) $(OLDLIB)

ifeq ($(OMP),OFF)
	echo "$(STATICLIB) and $(DYNAMICLIB) and $(LEGLIB)  and $(OLDLIB) built, single-thread versions"
else
	echo "$(STATICLIB) and $(DYNAMICLIB) and $(LEGLIB)  and $(OLDLIB) built, multithreaded versions"
endif

$(STATICLIB): $(OBJS) $(HEADERS)
	ar rcs $(STATICLIB) $(OBJS) 
$(DYNAMICLIB): $(OBJS) $(HEADERS)
	$(CXX) -shared $(OMPFLAGS) $(OBJS)  -o $(DYNAMICLIB) $(LIBSFFT)
$(LEGLIB): $(LEG_OBJS) $(OBJS)  $(HEADERS)
	ar rcs $(LEGLIB) $(LEG_OBJS) $(OBJS)  
$(OLDLIB): $(OLD_OBJS) $(OLD_HEADERS)
	ar rcs $(OLDLIB) $(OLD_OBJS) 

# here $(OMPFLAGS) and $(LIBSFFT) is needed for mac osx.
# see: http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html
# Also note -l libs come after objects, as per modern GCC requirement.

# examples in C++ and C... (separate codes for double vs single prec)
EX=examples/example1d1$(PRECSUFFIX)
EXC=examples/example1d1c$(PRECSUFFIX)
EX2=examples/example2d1

examples: $(EX) $(EXC) $(EX2)
	./$(EX)
	./$(EXC)
	./$(EX2)

$(EX): $(EX).o $(LEGLIB)
	$(CXX) $(CXXFLAGS) $(EX).o $(LEGLIB) $(LIBSFFT) -o $(EX)
$(EX2): $(EX2).o $(LEGLIB)
	$(CXX) $(CXXFLAGS) $(EX2).o $(LEGLIB) $(LIBSFFT) -o $(EX2)
$(EXC): $(EXC).o $(LEGLIB)
	$(CC) $(CFLAGS) $(EXC).o $(LEGLIB) $(LIBSFFT) $(CLINK) -o $(EXC)

# validation tests... (most link to .o allowing testing pieces separately)

test: $(LEG_STATICLIB) test/finufft1d_test test/finufft2d_test test/finufft3d_test test/dumbinputs test/finufft3dmany_test test/finufft2dmany_test  test/finufft1dmany_test test/finufftGuru_test test/dumbInputsGuru test/finufft1d_basicpassfail
	(cd test; \
	./finufft1d_basicpassfail \
	export FINUFFT_REQ_TOL=$(REQ_TOL); \
	export FINUFFT_CHECK_TOL=$(CHECK_TOL); \
	./check_finufft.sh)

test/finufft1d_basicpassfail: test/finufft1d_basicpassfail.cpp $(LEG_OBJS1)  $(OBJS)  $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft1d_basicpassfail.cpp $(LEG_OBJS1)  $(OBJS) $(LIBSFFT) -o test/finufft1d_basicpassfail

#test/testutils: test/testutils.cpp src/utils.o  $(HEADERS)
	$(CXX) $(CXXFLAGS) test/testutils.cpp src/utils.o -o test/testutils
test/finufft1d_test: test/finufft1d_test.cpp  $(LEG_OBJS1) $(OLD_OBJS1) $(OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft1d_test.cpp $(LEG_OBJS1) $(OLD_OBJS1) $(OBJS) $(LIBSFFT) -o test/finufft1d_test
test/finufft2d_test: test/finufft2d_test.cpp $(LEG_OBJS2) $(OBJS) $(OLD_OBJS2) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft2d_test.cpp $(LEG_OBJS2) $(OBJS) $(OLD_OBJS2) $(LIBSFFT) -o test/finufft2d_test
test/finufft3d_test: test/finufft3d_test.cpp $(LEG_OBJS3) $(OBJS) $(OLD_OBJS3) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft3d_test.cpp $(LEG_OBJS3) $(OBJS) $(OLD_OBJS3) $(LIBSFFT) -o test/finufft3d_test
test/dumbinputs: test/dumbinputs.cpp $(LEGLIB) $(OLD_OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/dumbinputs.cpp $(LEGLIB) $(OLD_OBJS) $(LIBSFFT) -o test/dumbinputs
test/dumbInputsGuru: test/dumbInputsGuru.cpp $(LEG_OBJS) $(OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/dumbInputsGuru.cpp $(LEG_OBJS) $(OBJS) $(LIBSFFT) -o test/dumbInputsGuru
test/finufft3dmany_test: test/finufft3dmany_test.cpp $(LEG_OBJS3) $(OBJS) $(OLD_OBJS3) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft3dmany_test.cpp $(LEG_OBJS3) $(OBJS) $(OLD_OBJS3) $(LIBSFFT) -o test/finufft3dmany_test
test/finufft2dmany_test: test/finufft2dmany_test.cpp $(LEG_OBJS2) $(OBJS) $(OLD_OBJS2) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft2dmany_test.cpp $(LEG_OBJS2) $(OBJS) $(OLD_OBJS2) $(LIBSFFT) -o test/finufft2dmany_test
test/finufft1dmany_test: test/finufft1dmany_test.cpp $(LEG_OBJS1) $(OBJS) $(OLD_OBJS1) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufft1dmany_test.cpp $(LEG_OBJS1) $(OBJS) $(OLD_OBJS1) $(LIBSFFT) -o test/finufft1dmany_test
test/finufftGuru_test: test/finufftGuru_test.cpp test/runOldFinufft.o $(OLD_OBJS) $(OBJS)  $(HEADERS)
	$(CXX) $(CXXFLAGS) test/finufftGuru_test.cpp test/runOldFinufft.o $(OLD_OBJS) $(OBJS) $(LIBSFFT) -o test/finufftGuru_test


# performance tests...
perftest: test/spreadtestnd test/finufft1d_test test/finufft2d_test test/finufft3d_test
# here the tee cmd copies output to screen. 2>&1 grabs both stdout and stderr...
	(cd test; ./spreadtestnd.sh 2>&1 | tee results/spreadtestnd_results.txt)
	(cd test; ./nuffttestnd.sh 2>&1 | tee results/nuffttestnd_results.txt)
test/spreadtestnd: test/spreadtestnd.cpp $(SOBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) test/spreadtestnd.cpp $(SOBJS) $(LIBS) -o test/spreadtestnd

# spreader only test (useful for development work on spreader)...
spreadtest: test/spreadtestnd
	test/spreadtestnd 1 8e6 8e6 1e-6 1 0
	test/spreadtestnd 2 8e6 8e6 1e-6 1 0
	test/spreadtestnd 3 8e6 8e6 1e-6 1 0

# --------------- LANGUAGE INTERFACES -----------------------
# fortran interface...
F1=fortran/nufft1d_demo$(PRECSUFFIX)
F2=fortran/nufft2d_demo$(PRECSUFFIX)
F3=fortran/nufft3d_demo$(PRECSUFFIX)
F4=fortran/nufft2dmany_demo$(PRECSUFFIX)
fortran: $(FOBJS) $(OBJS) $(HEADERS)
	$(FC) $(FFLAGS) $(F1).f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o $(F1)
	$(FC) $(FFLAGS) $(F2).f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o $(F2)
	$(FC) $(FFLAGS) $(F3).f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o $(F3)
	$(FC) $(FFLAGS) $(F4).f $(FOBJS) $(OBJS) $(LIBSFFT) $(FLINK) -o $(F4)
	time -p $(F1)
	time -p $(F2)
	time -p $(F3)
	time -p $(F4)

# matlab .mex* executable...
matlab: $(STATICLIB) $(HEADERS) matlab/finufft_m.cpp
ifeq ($(PREC),SINGLE)
	@echo "MATLAB interface only supports double precision; doing nothing"
else
	$(MEX) matlab/finufft.cpp $(STATICLIB) matlab/finufft_m.cpp $(MFLAGS) $(LIBSFFT) -output matlab/finufft
endif

# octave .mex executable... (also creates matlab/finufft.o for some reason)
octave: $(STATICLIB) $(HEADERS) matlab/finufft_m.cpp
ifeq ($(PREC),SINGLE)
	@echo "Octave interface only supports double precision; doing nothing"
else
	(cd matlab; mkoctfile --mex finufft.cpp ../$(STATICLIB) finufft_m.cpp $(OFLAGS) $(LIBSFFT) -output finufft)
	@echo "Running octave interface test; please wait a few seconds..."
	(cd matlab; octave check_finufft.m)
endif

# for experts; force rebuilds fresh MEX (matlab/octave) gateway via mwrap...
# (needs mwrap)
mex: matlab/finufft.mw
	(cd matlab;\
	$(MWRAP) -list -mex finufft -cppcomplex -mb finufft.mw ;\
	$(MWRAP) -mex finufft -c finufft.cpp -cppcomplex finufft.mw )

# python(3) interfaces...
python: $(STATICLIB)
ifeq ($(PREC),SINGLE)
	@echo "python interface only supports double precision; doing nothing"
else
	pip install .
	python python_tests/demo1d1.py
	python python_tests/run_accuracy_tests.py
endif
python3: $(STATICLIB)
ifeq ($(PREC),SINGLE)
	@echo "python3 interface only supports double precision; doing nothing"
else
	pip3 install .
	python3 python_tests/demo1d1.py
	python3 python_tests/run_accuracy_tests.py
endif


# ------------- Various obscure/devel tests -----------------
# This was for a CCQ application; zgemm was 10x faster!
test/manysmallprobs: $(STATICLIB) $(HEADERS) test/manysmallprobs.cpp
	$(CXX) $(CXXFLAGS) test/manysmallprobs.cpp $(STATICLIB) -o test/manysmallprobs $(LIBSFFT)
	(export OMP_NUM_THREADS=1; time test/manysmallprobs; unset OMP_NUM_THREADS)


# ------------- Cleaning up (including all versions of lib, and interfaces)...
clean: objclean pyclean
	rm -f lib-static/*.a lib/*.so
	rm -f matlab/*.mex*
	rm -f test/spreadtestnd test/finufft?d_test test/finufft?d_test test/testutils test/manysmallprobs test/results/*.out fortran/*_demo fortran/*_demof examples/example1d1 examples/example1d1c examples/example1d1f examples/example1d1cf test/finufftGuru1_test test/finufftGuru2_test test/dumbInputsGuru

# this is needed before changing precision or threading...
objclean:
	rm -f $(OBJS) $(OLD_OBJS) $(LEG_OBJS)
	rm -f fortran/*.o examples/*.o matlab/*.o

pyclean:
	rm -f finufftpy/*.pyc finufftpy/__pycache__/* python_tests/*.pyc python_tests/__pycache__/*

# for experts; only do this if you have mwrap to rebuild the interfaces!
mexclean:
	rm -f matlab/finufft.cpp matlab/finufft?d?.m matlab/finufft?d?many.m matlab/finufft.mex*
