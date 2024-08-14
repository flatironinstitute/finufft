# Makefile for FINUFFT (CPU code only, and its various interfaces)

# For simplicity, this is the only makefile; there are no makefiles in
# subdirectories. This makefile is also useful to show humans how to compile
# FINUFFT and its various language interfaces and examples.
# Users should not need to edit this makefile (doing so would make it hard to
# stay up to date with the repo version). Rather, in order to change
# OS/environment-specific compilers and flags, create the file make.inc, which
# overrides the defaults below (which are for an ubuntu linux/GCC system).
# See docs/install.rst, and make.inc.* for examples.

# Barnett 2017-2020. Malleo's expansion for guru interface, summer 2019.
# Barnett tidying Feb, May 2020. Libin Lu edits, 2020.
# Garrett Wright, Joakim Anden, Barnett: dual-prec lib build, Jun-Jul'20.
# Windows compatibility, jonas-kr, Sep '20.
# XSIMD dependency, Marco Barbone, June 2024.
# DUCC optional dependency to replace FFTW3. Barnett/Lu, 8/6/24.

# Compiler (CXX), and linking from C, fortran. We use GCC by default...
CXX = g++
CC = gcc
FC = gfortran
CLINK = -lstdc++
FLINK = $(CLINK)
PYTHON = python3
# baseline compile flags for GCC (no multithreading):
# Notes: 1) -Ofast breaks isfinite() & isnan(), so use -O3 which now is as fast
#        2) -fcx-limited-range for fortran-speed complex arith in C++
#        3) we use simply-expanded (:=) makefile variables, otherwise confusing
#        4) the extra math flags are for speed, but they do not impact accuracy;
#           they allow gcc to vectorize the code more effectively
CFLAGS := -O3 -funroll-loops -march=native -fcx-limited-range -ffp-contract=fast\
		  -fno-math-errno -fno-signed-zeros -fno-trapping-math -fassociative-math\
		  -freciprocal-math -fmerge-all-constants -ftree-vectorize $(CFLAGS)
FFLAGS := $(CFLAGS) $(FFLAGS)
CXXFLAGS := $(CFLAGS) $(CXXFLAGS)
# FFTW base name, and math linking...
FFTWNAME = fftw3
# linux default is fftw3_omp, since 10% faster than fftw3_threads...
FFTWOMPSUFFIX = omp
LIBS := -lm
# multithreading for GCC: C++/C/Fortran, MATLAB, and octave (ICC differs)...
OMPFLAGS = -fopenmp
OMPLIBS = -lgomp
# we bundle any libs mex needs here with flags...
MOMPFLAGS = -D_OPENMP $(OMPLIBS)
OOMPFLAGS =
# MATLAB MEX compilation (also see below +=)...
MFLAGS := -DR2008OO -largeArrayDims
# location of MATLAB's mex compiler (could add flags to switch GCC, etc)...
MEX = mex
# octave, and its mkoctfile and flags (also see below +=)...
OCTAVE = octave
MKOCTFILE = mkoctfile
OFLAGS = -DR2008OO
# For experts only, location of MWrap executable (see docs/install.rst):
MWRAP = mwrap

# root directory for dependencies to be downloaded:
DEPS_ROOT := deps

# xsimd header-only dependency repo
XSIMD_URL := https://github.com/xtensor-stack/xsimd.git
XSIMD_VERSION := 13.0.0
XSIMD_DIR := $(DEPS_ROOT)/xsimd

# DUCC sources optional dependency repo
DUCC_URL := https://gitlab.mpcdf.mpg.de/mtr/ducc.git
DUCC_VERSION := ducc0_0_34_0
DUCC_DIR := $(DEPS_ROOT)/ducc
# this dummy file used as empty target by make...
DUCC_COOKIE := $(DUCC_DIR)/.finufft_has_ducc
# for internal DUCC compile...
DUCC_INCL := -I$(DUCC_DIR)/src
DUCC_SRC := $(DUCC_DIR)/src/ducc0
# for DUCC objects compile only (not our objects)...  *** check flags, pthreads?:
DUCC_CXXFLAGS := -fPIC -std=c++17 -ffast-math

# absolute path of this makefile, ie FINUFFT's top-level directory...
FINUFFT = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# For your OS, override the above by setting make variables in make.inc ...
# (Please look in make.inc.* for ideas)
-include make.inc

# Now come flags that should be added, whatever user overrode in make.inc.
# -fPIC (position-indep code) needed to build dyn lib (.so)
# Also, we force return (via :=) to the land of simply-expanded variables...
INCL = -Iinclude -I$(XSIMD_DIR)/include
# single-thread total list of math and FFT libs (now both precisions)...
# (Note: finufft tests use LIBSFFT; spread & util tests only need LIBS)
LIBSFFT := $(LIBS)
ifeq ($(FFT),DUCC)
  DUCC_SETUP := $(DUCC_COOKIE)
# so FINUFFT build can see DUCC headers...
  INCL += $(DUCC_INCL)
  DUCC_OBJS := $(DUCC_SRC)/infra/string_utils.o $(DUCC_SRC)/infra/threading.o $(DUCC_SRC)/infra/mav.o $(DUCC_SRC)/math/gridding_kernel.o $(DUCC_SRC)/math/gl_integrator.o
  DUCC_SRCS := $(DUCC_OBJS:.o=.cc)
# FINUFFT's switchable FFT done via this compile directive...
  CXXFLAGS += -DFINUFFT_USE_DUCC0
else
# link against FFTW3 single-threaded (leaves DUCC_OBJS and DUCC_SETUP undef)
  LIBSFFT += -l$(FFTWNAME) -l$(FFTWNAME)f
endif
CXXFLAGS := $(CXXFLAGS) $(INCL) -fPIC -std=c++17
CFLAGS := $(CFLAGS) $(INCL) -fPIC
# here /usr/include needed for fftw3.f "fortran header"... (JiriK: no longer)
FFLAGS := $(FFLAGS) $(INCL) -I/usr/include -fPIC

# multi-threaded libs & flags, and req'd flags (OO for new interface)...
ifneq ($(OMP),OFF)
  CXXFLAGS += $(OMPFLAGS)
  CFLAGS += $(OMPFLAGS)
  FFLAGS += $(OMPFLAGS)
  MFLAGS += $(MOMPFLAGS)
  OFLAGS += $(OOMPFLAGS)
  LIBS += $(OMPLIBS)
# fftw3 multithreaded libs...
  ifneq ($(FFT),DUCC)
    LIBSFFT += -l$(FFTWNAME)_$(FFTWOMPSUFFIX) -l$(FFTWNAME)f_$(FFTWOMPSUFFIX) $(OMPLIBS)
  endif
endif

# name & location of shared library we're building...
LIBNAME = libfinufft
ifeq ($(MINGW),ON)
  DYNLIB = lib/$(LIBNAME).dll
else
  DYNLIB = lib/$(LIBNAME).so
endif

STATICLIB = lib-static/$(LIBNAME).a
# absolute path to the .so, useful for linking so executables portable...
ABSDYNLIB = $(FINUFFT)$(DYNLIB)

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
OBJS = $(SOBJS) src/finufft.o src/simpleinterfaces.o fortran/finufftfort.o src/fft.o
# their single-prec versions
OBJSF = $(OBJS:%.o=%_32.o)
# precision-dependent library object files (compiled & linked only once)...
OBJS_PI = $(SOBJS_PI) contrib/legendre_rule_fast.o
# all lib dual-precision objs (note DUCC_OBJS empty if unused)
OBJSD = $(OBJS) $(OBJSF) $(OBJS_PI) $(DUCC_OBJS)

.PHONY: usage lib examples test perftest spreadtest spreadtestall fortran matlab octave all mex python clean objclean pyclean mexclean wheel docker-wheel gurutime docs setup setupclean

default: usage

all: test perftest lib examples fortran matlab octave python

usage:
	@echo "Makefile for FINUFFT library. Please specify your task:"
	@echo " make lib - build the main library (in lib/ and lib-static/)"
	@echo " make examples - compile and run all codes in examples/"
	@echo " make test - compile and run quick math validation tests"
	@echo " make perftest - compile and run (slower) performance tests"
	@echo " make fortran - compile and run Fortran tests and examples"
	@echo " make matlab - compile MATLAB interfaces (no test)"
	@echo " make octave - compile and test octave interfaces"
	@echo " make python - compile and test python interfaces"
	@echo " make all - do all the above (around 1 minute; assumes you have MATLAB, etc)"
	@echo " make spreadtest - compile & run spreader-only tests (no FFT)"
	@echo " make spreadtestall - small set spreader-only tests for CI use"
	@echo " make objclean - remove all object files, preserving libs & MEX"
	@echo " make clean - also remove all lib, MEX, py, and demo executables"
	@echo " make setup - check (and possibly download) dependencies"
	@echo " make setupclean - delete downloaded dependencies"
	@echo "For faster (multicore) compilation, append, for example, -j8"
	@echo ""
	@echo "Make options:"
	@echo " 'make [task] OMP=OFF' for single-threaded (no refs to OpenMP)"
	@echo " 'make [task] FFT=DUCC' for DUCC0 FFT (otherwise uses FFTW3)"
	@echo " You must at least 'make objclean' before changing such options!"
	@echo ""
	@echo "Also see docs/install.rst and docs/README"

# collect headers for implicit depends (we don't separate public from private)
HEADERS = $(wildcard include/*.h include/finufft/*.h) $(DUCC_HEADERS)

# implicit rules for objects (note -o ensures writes to correct dir)
%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $< -o $@
%_32.o: %.cpp $(HEADERS)
	$(CXX) -DSINGLE -c $(CXXFLAGS) $< -o $@
%.o: %.c $(HEADERS)
	$(CC) -c $(CFLAGS) $< -o $@
%_32.o: %.c $(HEADERS)
	$(CC) -DSINGLE -c $(CFLAGS) $< -o $@
%.o: %.f
	$(FC) -c $(FFLAGS) $< -o $@
%_32.o: %.f
	$(FC) -DSINGLE -c $(FFLAGS) $< -o $@

# spreadinterp include auto-generated code, xsimd header-only dependency;
# if FFT=DUCC also setup ducc with fft.h dependency on $(DUCC_SETUP)...
# Note src/spreadinterp.cpp includes finufft/defs.h which includes finufft/fft.h
# so fftw/ducc header needed for spreadinterp, though spreadinterp should not
# depend on fftw/ducc directly?
include/finufft/fft.h: $(DUCC_SETUP)
SHEAD = $(wildcard src/*.h) $(XSIMD_DIR)/include/xsimd/xsimd.hpp
src/spreadinterp.o: $(SHEAD)
src/spreadinterp_32.o: $(SHEAD)


# lib -----------------------------------------------------------------------
# build library with double/single prec both bundled in...
lib: $(STATICLIB) $(DYNLIB)
$(STATICLIB): $(OBJSD)
	ar rcs $(STATICLIB) $(OBJSD)
ifeq ($(OMP),OFF)
	@echo "$(STATICLIB) built, single-thread version"
else
	@echo "$(STATICLIB) built, multithreaded version"
endif
$(DYNLIB): $(OBJSD)
# using *absolute* path in the -o here is needed to make portable executables
# when compiled against it, in mac OSX, strangely...
	$(CXX) -shared ${LDFLAGS} $(OMPFLAGS) $(OBJSD) -o $(ABSDYNLIB) $(LIBSFFT)
ifeq ($(OMP),OFF)
	@echo "$(DYNLIB) built, single-thread version"
else
	@echo "$(DYNLIB) built, multithreaded version"
endif

# here $(OMPFLAGS) and $(LIBSFFT) is even needed for linking under mac osx.
# see: http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html
# Also note -l libs come after objects, as per modern GCC requirement.


# examples (C++/C) -----------------------------------------------------------
# build all examples (single-prec codes separate, and not all have one)...
EXAMPLES := $(basename $(wildcard examples/*.c examples/*.cpp))
ifeq ($(OMP),OFF)
  EXAMPLES := $(filter-out $(basename $(wildcard examples/*thread*.cpp)),$(EXAMPLES))
endif
examples: $(EXAMPLES)
ifneq ($(MINGW),ON)
  # Windows-MSYS does not find the dynamic libraries, so we make a temporary copy
  # Windows-MSYS has same commands as Linux/OSX
  ifeq ($(MSYS),ON)
	cp $(DYNLIB) test
  endif
  # non-Windows-WSL: this task always runs them (note escaped $ to pass to bash)...
	for i in $(EXAMPLES); do echo $$i...; ./$$i; done
else
  # Windows-WSL does not find the dynamic libraries, so we make a temporary copy
	copy $(DYNLIB) examples
	for /f "delims= " %%i in ("$(subst /,\,$(EXAMPLES))") do (echo %%i & %%i.exe)
	del examples\$(LIBNAME).so
endif
	@echo "Done running: $(EXAMPLES)"
# fun fact: gnu make patterns match those with shortest "stem", so this works:
examples/%: examples/%.o $(DYNLIB)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} $< $(ABSDYNLIB) $(LIBSFFT) -o $@
examples/%c: examples/%c.o $(DYNLIB)
	$(CC) $(CFLAGS) ${LDFLAGS} $< $(ABSDYNLIB) $(LIBSFFT) $(CLINK) -o $@
examples/%cf: examples/%cf.o $(DYNLIB)
	$(CC) $(CFLAGS) ${LDFLAGS} $< $(ABSDYNLIB) $(LIBSFFT) $(CLINK) -o $@


# test (library validation) --------------------------------------------------
# build (skipping .o) but don't run. Run with 'test' target
# Note: both precisions use same sources; single-prec executables get f suffix.
# generic tests link against our .so... (other libs needed for fftw_forget...)
test/%: test/%.cpp $(DYNLIB)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} $< $(ABSDYNLIB) $(LIBSFFT) -o $@
test/%f: test/%.cpp $(DYNLIB)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} -DSINGLE $< $(ABSDYNLIB) $(LIBSFFT) -o $@
# low-level tests that are cleaner if depend on only specific objects...
test/testutils: test/testutils.cpp src/utils.o src/utils_precindep.o
	$(CXX) $(CXXFLAGS) ${LDFLAGS} test/testutils.cpp src/utils.o src/utils_precindep.o $(LIBS) -o test/testutils
test/testutilsf: test/testutils.cpp src/utils_32.o src/utils_precindep.o
	$(CXX) $(CXXFLAGS) ${LDFLAGS} -DSINGLE test/testutils.cpp src/utils_32.o src/utils_precindep.o $(LIBS) -o test/testutilsf

# make sure all double-prec test executables ready for testing
TESTS := $(basename $(wildcard test/*.cpp))
# also need single-prec
TESTS += $(TESTS:%=%f)
test: $(TESTS)
ifneq ($(MINGW),ON)
  # non-Windows-WSL: it will fail if either of these return nonzero exit code...
  # Windows-MSYS does not find the dynamic libraries, so we make a temporary copy
  # Windows-MSYS has same commands as Linux/OSX
  ifeq ($(MSYS),ON)
	cp $(DYNLIB) test
  endif
	test/basicpassfail
	test/basicpassfailf
  # accuracy tests done in prec-switchable bash script... (small prob -> few thr)
	(cd test; export OMP_NUM_THREADS=4; ./check_finufft.sh; ./check_finufft.sh SINGLE)
else
  # Windows-WSL does not find the dynamic libraries, so we make a temporary copy...
	copy $(DYNLIB) test
	test/basicpassfail
	test/basicpassfailf
  # Windows does not feature a bash shell so we use WSL. Since most supplied gnu-make variants are 32bit executables and WSL runs only in 64bit environments, we have to refer to 64bit powershell explicitly on 32bit make...
  #	$(windir)\Sysnative\WindowsPowerShell\v1.0\powershell.exe "cd ./test; bash check_finufft.sh DOUBLE $(MINGW); bash check_finufft.sh SINGLE $(MINGW)"
  # with a recent version of gnu-make for Windows built for 64bit as it is part of the WinLibs standalone build of GCC and MinGW-w64 we can avoid these circumstances
	cd test
	bash -c "cd test; ./check_finufft.sh DOUBLE $(MINGW)"
	bash -c "cd test; ./check_finufft.sh SINGLE $(MINGW)"
	del test\$(LIBNAME).so
endif


# perftest (performance/developer tests) -------------------------------------
# generic perf test rules...
perftest/%: perftest/%.cpp $(DYNLIB)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} $< $(ABSDYNLIB) $(LIBSFFT) -o $@
perftest/%f: perftest/%.cpp $(DYNLIB)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} -DSINGLE $< $(ABSDYNLIB) $(LIBSFFT) -o $@

# spreader only test, double/single (good for self-contained work on spreader)
ST=perftest/spreadtestnd
STA=perftest/spreadtestndall
STF=$(ST)f
STAF=$(STA)f
$(ST): $(ST).cpp $(SOBJS) $(SOBJS_PI)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} $< $(SOBJS) $(SOBJS_PI) $(LIBS) -o $@
$(STF): $(ST).cpp $(SOBJSF) $(SOBJS_PI)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} -DSINGLE $< $(SOBJSF) $(SOBJS_PI) $(LIBS) -o $@
$(STA): $(STA).cpp $(SOBJS) $(SOBJS_PI)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} $< $(SOBJS) $(SOBJS_PI) $(LIBS) -o $@
$(STAF): $(STA).cpp $(SOBJSF) $(SOBJS_PI)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} -DSINGLE $< $(SOBJSF) $(SOBJS_PI) $(LIBS) -o $@
spreadtest: $(ST) $(STF)
# run one thread per core... (escape the $ to get single $ in bash; one big cmd)
	(export OMP_NUM_THREADS=$$(perftest/mynumcores.sh) ;\
	echo "\nRunning makefile double-precision spreader tests, $$OMP_NUM_THREADS threads..." ;\
	$(ST) 1 8e6 8e6 1e-6 ;\
	$(ST) 2 8e6 8e6 1e-6 ;\
	$(ST) 3 8e6 8e6 1e-6 ;\
	echo "\nRunning makefile single-precision spreader tests, $$OMP_NUM_THREADS threads..." ;\
	$(STF) 1 8e6 8e6 1e-3 ;\
	$(STF) 2 8e6 8e6 1e-3 ;\
	$(STF) 3 8e6 8e6 1e-3 )
# smaller test of spreadinterp various tols, precs, kermeths...
spreadtestall: $(ST) $(STF)
	(cd perftest; ./spreadtestall.sh)
# Marco's sweep through kernel widths (ie tols)...
spreadtestndall: $(STA) $(STAF)
	(cd perftest; ./multispreadtestndall.sh)
bigtest: perftest/big2d2f
	@echo "\nRunning >2^31 size example (takes 30 s and 30 GB RAM)..."
	perftest/big2d2f

PERFEXECS := $(basename $(wildcard test/finufft?d_test.cpp))
PERFEXECS += $(PERFEXECS:%=%f)
perftest: $(ST) $(STF) $(PERFEXECS) spreadtestndall bigtest
# here the tee cmd copies output to screen. 2>&1 grabs both stdout and stderr...
	(cd perftest ;\
	./spreadtestnd.sh 2>&1 | tee results/spreadtestnd_results.txt ;\
	./spreadtestnd.sh SINGLE 2>&1 | tee results/spreadtestndf_results.txt ;\
	./nuffttestnd.sh 2>&1 | tee results/nuffttestnd_results.txt ;\
	./nuffttestnd.sh SINGLE 2>&1 | tee results/nuffttestndf_results.txt )

# speed ratio of many-vector guru vs repeated single calls... (Andrea)
GTT=perftest/guru_timing_test
GTTF=$(GTT)f
gurutime: $(GTT) $(GTTF)
	for i in $(GTT) $(GTTF); do $$i 100 1 2 1e2 1e2 0 1e6 1e-3 1 0 0 2; done

# This was for a CCQ application... (zgemm was 10x faster! double-prec only)
perftest/manysmallprobs: perftest/manysmallprobs.cpp $(STATICLIB)
	$(CXX) $(CXXFLAGS) ${LDFLAGS} $< $(STATICLIB) $(LIBSFFT) -o $@
	@echo "manysmallprobs: single-thread..."
	OMP_NUM_THREADS=1 $@




# ======================= LANGUAGE INTERFACES ==============================

# fortran --------------------------------------------------------------------
FD = fortran/directft
# CMCL NUFFT fortran test codes (only needed by the nufft*_demo* codes)
CMCLOBJS = $(FD)/dirft1d.o $(FD)/dirft2d.o $(FD)/dirft3d.o $(FD)/dirft1df.o $(FD)/dirft2df.o $(FD)/dirft3df.o $(FD)/prini.o
# build examples list...
FE_DIR = fortran/examples
FE64 = $(FE_DIR)/simple1d1 $(FE_DIR)/simple1d1_f90 $(FE_DIR)/guru1d1 $(FE_DIR)/nufft1d_demo $(FE_DIR)/nufft2d_demo $(FE_DIR)/nufft3d_demo $(FE_DIR)/nufft2dmany_demo
# add the "f" single-prec suffix to all examples except the f90 one...
FE32 := $(filter-out %/simple1d1_f90f, $(FE64:%=%f))
# list of all fortran examples
FE = $(FE64) $(FE32)

# fortran target pattern match (no longer runs executables)
$(FE_DIR)/%: $(FE_DIR)/%.f $(CMCLOBJS) $(DYNLIB)
	$(FC) $(FFLAGS) ${LDFLAGS} $< $(CMCLOBJS) $(ABSDYNLIB) $(FLINK) -o $@
$(FE_DIR)/%f: $(FE_DIR)/%f.f $(CMCLOBJS) $(DYNLIB)
	$(FC) $(FFLAGS) ${LDFLAGS} $< $(CMCLOBJS) $(ABSDYNLIB) $(FLINK) -o $@
# fortran90 lone demo
$(FE_DIR)/simple1d1_f90: $(FE_DIR)/simple1d1.f90 include/finufft_mod.f90 $(CMCLOBJS) $(DYNLIB)
	$(FC) $(FFLAGS) ${LDFLAGS} include/finufft_mod.f90 $< $(CMCLOBJS) $(ABSDYNLIB) $(FLINK) -o $@

fortran: $(FE)
# this task runs them (note escaped $ to pass to bash)...
	for i in $(FE); do echo $$i...; ./$$i; done
	@echo "Done running: $(FE)"


# matlab ----------------------------------------------------------------------
# matlab .mex* executable... (matlab is so slow to start, not worth testing it)
matlab: matlab/finufft.cpp $(STATICLIB)
	$(MEX) $< $(STATICLIB) $(INCL) $(MFLAGS) $(LIBSFFT) -output matlab/finufft

# octave .mex executable...
octave: matlab/finufft.cpp $(STATICLIB)
	(cd matlab; $(MKOCTFILE) --mex finufft.cpp -I../include ../$(STATICLIB) $(OFLAGS) $(LIBSFFT) -output finufft)
	@echo "Running octave interface tests; please wait a few seconds..."
	(cd matlab ;\
	$(OCTAVE) test/check_finufft.m ;\
	$(OCTAVE) test/check_finufft_single.m ;\
	$(OCTAVE) examples/guru1d1.m ;\
	$(OCTAVE) examples/guru1d1_single.m)

# for experts: force rebuilds fresh MEX (matlab/octave) gateway
# matlab/finufft.cpp via mwrap (needs recent version of mwrap >= 0.33.10)...
mex: matlab/finufft.mw
ifneq ($(MINGW),ON)
	(cd matlab ;\
	$(MWRAP) -mex finufft -c finufft.cpp -mb -cppcomplex finufft.mw)
else
	(cd matlab & $(MWRAP) -mex finufft -c finufft.cpp -mb -cppcomplex finufft.mw)
endif


# python ---------------------------------------------------------------------
python: $(STATICLIB) $(DYNLIB)
	FINUFFT_DIR=$(FINUFFT) $(PYTHON) -m pip -v install python/finufft
# note to devs: if trouble w/ NumPy, use: pip install ./python --no-deps
	$(PYTHON) python/finufft/test/run_accuracy_tests.py
	$(PYTHON) python/finufft/examples/simple1d1.py
	$(PYTHON) python/finufft/examples/simpleopts1d1.py
	$(PYTHON) python/finufft/examples/guru1d1.py
	$(PYTHON) python/finufft/examples/guru1d1f.py
	$(PYTHON) python/finufft/examples/simple2d1.py
	$(PYTHON) python/finufft/examples/many2d1.py
	$(PYTHON) python/finufft/examples/guru2d1.py
	$(PYTHON) python/finufft/examples/guru2d1f.py

# general python packaging wheel for all OSs without wheel being fixed(required shared libs are not included in wheel)
python-dist: $(STATICLIB) $(DYNLIB)
	(export FINUFFT_DIR=$(shell pwd); cd python/finufft; $(PYTHON) -m pip wheel . -w wheelhouse)

# python packaging wheel for macosx with wheel being fixed(all required shared libs are included in wheel)
wheel: $(STATICLIB) $(DYNLIB)
	(export FINUFFT_DIR=$(shell pwd); cd python/finufft; $(PYTHON) -m pip wheel . -w wheelhouse; delocate-wheel -w fixed_wheel -v wheelhouse/finufft*.whl)

docker-wheel:
	docker run --rm -e package_name=finufft -v `pwd`:/io libinlu/manylinux2010_x86_64_fftw /io/python/ci/build-wheels.sh


# ================== SETUP/COMPILE OF EXTERNAL DEPENDENCIES ===============

define clone_repo
    @if [ ! -d "$(3)" ]; then \
        echo "Cloning repository $(1) at tag $(2) into directory $(3)"; \
        git clone --depth=1 --branch $(2) $(1) $(3); \
    else \
        cd $(3) && \
        CURRENT_VERSION=$$(git describe --tags --abbrev=0) && \
        if [ "$$CURRENT_VERSION" = "$(2)" ]; then \
            echo "Directory $(3) already exists and is at the correct version $(2)."; \
        else \
            echo "Directory $(3) exists but is at version $$CURRENT_VERSION. Checking out the correct version $(2)."; \
            git fetch --tags && \
            git checkout $(2) || { echo "Error: Failed to checkout version $(2) in $(3)."; exit 1; }; \
        fi; \
    fi
endef

# download: header-only, no compile needed...
$(XSIMD_DIR)/include/xsimd/xsimd.hpp:
	mkdir -p $(DEPS_ROOT)
	@echo "Checking XSIMD external dependency..."
	$(call clone_repo,$(XSIMD_URL),$(XSIMD_VERSION),$(XSIMD_DIR))
	@echo "xsimd installed in deps/xsimd"

# download DUCC... (an empty target just used to track if installed)
$(DUCC_COOKIE):
	mkdir -p $(DEPS_ROOT)
	@echo "Checking DUCC external dependency..."
	$(call clone_repo,$(DUCC_URL),$(DUCC_VERSION),$(DUCC_DIR))
	touch $(DUCC_COOKIE)
	@echo "DUCC installed in deps/ducc"

# implicit rule for DUCC compile just needed objects, only used if FFT=DUCC.
# Needed since DUCC has no makefile (yet).
$(DUCC_SRCS): %.cc: $(DUCC_SETUP)
$(DUCC_OBJS): %.o: %.cc
	$(CXX) -c $(DUCC_CXXFLAGS) $(DUCC_INCL) $< -o $@

setup: $(XSIMD_DIR)/include/xsimd/xsimd.hpp $(DUCC_SETUP)

setupclean:
	rm -rf $(DEPS_ROOT)


# =============================== DOCUMENTATION =============================

docs: docs/*.docsrc docs/matlabhelp.doc docs/makecdocs.sh
	(cd docs; ./makecdocs.sh)
# get the makefile help strings from make w/o args, stdout...
	make 1> docs/makefile.doc
docs/matlabhelp.doc: docs/genmatlabhelp.sh matlab/*.sh matlab/*.docsrc matlab/*.docbit matlab/*.m
	(cd matlab; ./addmhelp.sh)
	(cd docs; ./genmatlabhelp.sh)



# =============================== CLEAN UP ==================================

clean: objclean pyclean
ifneq ($(MINGW),ON)
  # non-Windows-WSL clean up...
	rm -f $(STATICLIB) $(DYNLIB)
	rm -f matlab/*.mex*
	rm -f $(TESTS) test/results/*.out perftest/results/*.out
	rm -f $(EXAMPLES) $(FE) $(ST) $(STF) $(STA) $(STAF) $(GTT) $(GTTF)
	rm -f perftest/manysmallprobs perftest/big2d2f
	rm -f examples/core test/core perftest/core $(FE_DIR)/core
else
  # Windows-WSL clean up...
	del $(subst /,\,$(STATICLIB)), $(subst /,\,$(DYNLIB))
	del matlab\*.mex*
	for %%f in ($(subst /,\, $(TESTS))) do ((if exist %%f del %%f) & (if exist %%f.exe del %%f.exe))
	del test\results\*.out perftest\results\*.out
	for %%f in ($(subst /,\, $(EXAMPLES)), $(subst /,\,$(FE)), $(subst /,\,$(ST)), $(subst /,\,$(STF)), $(subst /,\,$(STA)), $(subst /,\,$(STAF)), $(subst /,\,$(GTT)), $(subst /,\,$(GTTF))) do ((if exist %%f del %%f) & (if exist %%f.exe del %%f.exe))
	del perftest\manysmallprobs, perftest\big2d2f
	del examples\core, test\core, perftest\core, $(subst /,\, $(FE_DIR))\core
endif


# indiscriminate .o killer; needed before changing threading...
objclean:
ifneq ($(MINGW),ON)
  # non-Windows-WSL... (note: cleans DUCC objects regardless of FFT choice)
	rm -f src/*.o test/directft/*.o test/*.o examples/*.o matlab/*.o contrib/*.o
	rm -f fortran/*.o $(FE_DIR)/*.o $(FD)/*.o finufft_mod.mod
	rm -f $(DUCC_SRC)/infra/*.o $(DUCC_SRC)/math/*.o
else
  # Windows-WSL...
	for /d %%d in (src,test\directfttest,examples,matlab,contrib) do (for %%f in (%%d\*.o) do (del %%f))
	for /d %%d in (fortran,$(subst /,\, $(FE_DIR)),$(subst /,\, $(FD))) do (for %%f in (%%d\*.o) do (del %%f))
  # *** to del DUCC *.o
endif

pyclean:
ifneq ($(MINGW),ON)
  # non-Windows-WSL...
	rm -f python/finufft/*.pyc python/finufft/__pycache__/* python/test/*.pyc python/test/__pycache__/*
	rm -rf python/fixed_wheel python/wheelhouse
else
  # Windows-WSL...
	for /d %%d in (python\finufft,python\test) do (for %%f in (%%d\*.pyc) do (del %%f))
	for /d %%d in (python\finufft\__pycache__,python\test\__pycache__) do (for %%f in (%%d\*) do (del %%f))
	for /d %%d in (python\fixed_wheel,python\wheelhouse) do (if exist %%d (rmdir /s /q %%d))
endif

# for experts; only run this if you possess mwrap to rebuild the interfaces!
mexclean:
ifneq ($(MINGW),ON)
  # non-Windows-WSL...
	rm -f matlab/finufft_plan.m matlab/finufft.cpp matlab/finufft.mex*
else
  # Windows-WSL...
	del matlab\finufft_plan.m matlab\finufft.cpp matlab\finufft.mex*
endif
