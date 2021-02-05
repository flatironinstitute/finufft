# CUFINUFFT Makefile

# Load site-specific setting -- controlled using environment variable `site`:
# eg.  make site=nersc_cori
ifdef site
    $(info detected site: $(site))
    -include sites/make.inc.$(site)
endif

# Load architecture-specific settings -- controlled using the environment
# variable `target`: eg. make target=power9
ifdef target
    $(info detected target: $(target))
    -include targets/make.inc.$(target)
endif


CC   ?= gcc
CXX  ?= g++
NVCC ?= nvcc

# Developer-users are suggested to optimize NVARCH in their own make.inc, see:
#   http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
NVARCH ?= -arch=sm_70 \
	  -gencode=arch=compute_35,code=sm_35 \
	  -gencode=arch=compute_50,code=sm_50 \
	  -gencode=arch=compute_52,code=sm_52 \
	  -gencode=arch=compute_60,code=sm_60 \
	  -gencode=arch=compute_61,code=sm_61 \
	  -gencode=arch=compute_70,code=sm_70 \
	  -gencode=arch=compute_75,code=sm_75 \
	  -gencode=arch=compute_75,code=compute_75

CFLAGS    ?= -fPIC -O3 -funroll-loops -march=native
CXXFLAGS  ?= $(CFLAGS) -std=c++14
NVCCFLAGS ?= -std=c++14 -ccbin=$(CXX) -O3 $(NVARCH) -Wno-deprecated-gpu-targets \
	     --default-stream per-thread -Xcompiler "$(CXXFLAGS)"

# For debugging, tell nvcc to add symbols to host and device code respectively,
#NVCCFLAGS+= -g -G
# and enable cufinufft internal flags.
#NVCCFLAGS+= -DINFO -DDEBUG -DRESULT -DTIME

# CUDA Related build dependencies -- the user can overwrite CUDA_ROOT using the
# CUDA_DIR environment variable. If neither (CUDA_ROOT, nor CUDA_DIR) is set,
# CUDA_ROOT defaults to /usr/local/cuda
ifeq ($(CUDA_DIR),)
    CUDA_ROOT ?= /usr/local/cuda
else
    CUDA_ROOT := $(CUDA_DIR)
endif

# Common includes
INC += -I$(CUDA_ROOT)/include -Icontrib/cuda_samples

# NVCC-specific libs
NVCC_LIBS_PATH += -L$(CUDA_ROOT)/lib64
ifdef FFTW_DIR
    NVCC_LIBS_PATH += -L$(FFTW_DIR)
endif
ifdef NVCC_STUBS
    $(info detected CUDA_STUBS -- setting CUDA stubs directory)
    NVCC_LIBS_PATH += -L$(NVCC_STUBS)
endif

LIBS += -lm -lcudart -lstdc++ -lnvToolsExt -lcufft -lcuda


#############################################################
# Allow the user to override any variable above this point. #
-include make.inc

# Include header files
INC += -I include

LIBNAME=libcufinufft
DYNAMICLIB=lib/$(LIBNAME).so
STATICLIB=lib-static/$(LIBNAME).a

BINDIR=bin

HEADERS = include/cufinufft.h src/cudeconvolve.h src/memtransfer.h include/profile.h \
	src/cuspreadinterp.h include/cufinufft_eitherprec.h include/cufinufft_errors.h
CONTRIBOBJS=contrib/dirft2d.o contrib/common.o contrib/spreadinterp.o contrib/utils_fp.o

# We create three collections of objects:
#  Double (_64), Single (_32), and floating point agnostic (no suffix)

CUFINUFFTOBJS=src/precision_independent.o src/profile.o contrib/legendre_rule_fast.o contrib/utils.o
CUFINUFFTOBJS_64=src/2d/spreadinterp2d.o src/2d/cufinufft2d.o \
	src/2d/spread2d_wrapper.o src/2d/spread2d_wrapper_paul.o \
	src/2d/interp2d_wrapper.o src/memtransfer_wrapper.o \
	src/deconvolve_wrapper.o src/cufinufft.o \
	src/3d/spreadinterp3d.o src/3d/spread3d_wrapper.o \
	src/3d/interp3d_wrapper.o src/3d/cufinufft3d.o \
	$(CONTRIBOBJS)
CUFINUFFTOBJS_32=$(CUFINUFFTOBJS_64:%.o=%_32.o)


%_32.o: %.cpp $(HEADERS)
	$(CXX) -DSINGLE -c $(CXXFLAGS) $(INC) $< -o $@
%_32.o: %.c $(HEADERS)
	$(CC) -DSINGLE -c $(CFLAGS) $(INC) $< -o $@
%_32.o: %.cu $(HEADERS)
	$(NVCC) -DSINGLE --device-c -c $(NVCCFLAGS) $(INC) $< -o $@
%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.c $(HEADERS)
	$(CC) -c $(CFLAGS) $(INC) $< -o $@
%.o: %.cu $(HEADERS)
	$(NVCC) --device-c -c $(NVCCFLAGS) $(INC) $< -o $@

default: all

# Build all, but run no tests. Note: CI currently uses this default...
all: libtest spreadtest examples

# testers for the lib (does not execute)
libtest: lib $(BINDIR)/cufinufft2d1_test \
	$(BINDIR)/cufinufft2d2_test \
	$(BINDIR)/cufinufft2d1many_test \
	$(BINDIR)/cufinufft2d2many_test \
	$(BINDIR)/cufinufft2d1nupts_test \
	$(BINDIR)/cufinufft2d1_test_32 \
	$(BINDIR)/cufinufft2d2_test_32 \
	$(BINDIR)/cufinufft2d1many_test_32 \
	$(BINDIR)/cufinufft2d2many_test_32 \
	$(BINDIR)/cufinufft2d1nupts_test_32 \
	$(BINDIR)/cufinufft3d1_test \
	$(BINDIR)/cufinufft3d2_test \
	$(BINDIR)/cufinufft3d1_test_32 \
	$(BINDIR)/cufinufft3d2_test_32 \
	$(BINDIR)/cufinufft2d2api_test \
	$(BINDIR)/cufinufft2d2api_test_32

# low-level (not-library) testers (does not execute)
spreadtest: $(BINDIR)/spread2d_test \
	$(BINDIR)/spread2d_test_32 \
	$(BINDIR)/interp2d_test \
	$(BINDIR)/interp2d_test_32 \
	$(BINDIR)/spread3d_test \
	$(BINDIR)/spread3d_test_32 \
	$(BINDIR)/interp3d_test \
	$(BINDIR)/interp3d_test_32

examples: $(BINDIR)/example2d1many \
	$(BINDIR)/example2d2many

$(BINDIR)/example%: examples/example%.cpp $(DYNAMICLIB) $(HEADERS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIBS) -o $@ $< $(DYNAMICLIB)

$(BINDIR)/cufinufft2d2api_test%: test/cufinufft2d2api_test%.o $(DYNAMICLIB)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $< $(DYNAMICLIB)

$(BINDIR)/%_32: test/%_32.o $(CUFINUFFTOBJS_32) $(CUFINUFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) -DSINGLE $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@

$(BINDIR)/%: test/%.o $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@

# user-facing library...
lib: $(STATICLIB) $(DYNAMICLIB)

$(STATICLIB): $(CUFINUFFTOBJS) $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS_32) $(CONTRIBOBJS)
	mkdir -p lib-static
	ar rcs $(STATICLIB) $^
$(DYNAMICLIB): $(CUFINUFFTOBJS) $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS_32) $(CONTRIBOBJS)
	mkdir -p lib
	$(NVCC) -shared $(NVCCFLAGS) $^ -o $(DYNAMICLIB) $(LIBS)


# --------------------------------------------- start of check tasks ---------
# Check targets: in contrast to the above, these tasks just execute things:
check:
	@echo "Building lib, all testers, and running all tests..."
	$(MAKE) checkspread
	$(MAKE) checkapi
	$(MAKE) check2D
	$(MAKE) check3D
	$(MAKE) checkexamples

checkspread: spreadtest
	@echo "Running spread/interp only tests..."
	(cd test; ./spreadperf.sh)

checkapi: libtest
	@echo "Running API tests..."
	bin/cufinufft2d2api_test
	bin/cufinufft2d2api_test_32

check2D: check2D_64 check2D_32

# Note: we could kill the low-level spread/interp tests from here...
check2D_64: spreadtest libtest
	@echo Running 2-D cases
	bin/spread2d_test 1 1 16 16
	bin/spread2d_test 2 1 16 16
	bin/spread2d_test 1 1 1024 1024
	bin/spread2d_test 2 1 1024 1024
	bin/interp2d_test 1 1 16 16
	bin/interp2d_test 1 1 1024 1024
	bin/cufinufft2d1_test 1 8 8
	bin/cufinufft2d1_test 2 8 8
	bin/cufinufft2d1_test 1 256 256
	bin/cufinufft2d1_test 2 512 512
	bin/cufinufft2d2_test 1 8 8
	bin/cufinufft2d2_test 2 8 8
	bin/cufinufft2d2_test 1 256 256
	bin/cufinufft2d2_test 2 512 512
	@echo Running 2-D High Density cases
	bin/cufinufft2d1_test 1 64 64 8192
	bin/cufinufft2d1_test 2 64 64 8192
	bin/cufinufft2d2_test 1 64 64 8192
	bin/cufinufft2d2_test 2 64 64 8192
	@echo Running 2-D Low Density cases
	bin/cufinufft2d1_test 1 64 64 1024
	bin/cufinufft2d1_test 2 64 64 1024
	bin/cufinufft2d2_test 1 64 64 1024
	bin/cufinufft2d2_test 2 64 64 1024
	@echo Running 2-D-Many cases
	bin/cufinufft2d1many_test 1 64 64 128 1e-3
	bin/cufinufft2d1many_test 1 256 256 1024
	bin/cufinufft2d1many_test 2 512 512 256
	bin/cufinufft2d1many_test 1 1e2 2e2 3e2 16 1e4
	bin/cufinufft2d1many_test 2 1e2 2e2 3e2 16 1e4
	bin/cufinufft2d2many_test 1 64 64 128 1e-3
	bin/cufinufft2d2many_test 1 256 256 1024
	bin/cufinufft2d2many_test 2 512 512 256
	bin/cufinufft2d2many_test 1 256 256 1024
	bin/cufinufft2d2many_test 1 1e2 2e2 3e2 16 1e4
	bin/cufinufft2d2many_test 2 1e2 2e2 3e2 16 1e4

check2D_32: spreadtest libtest
	@echo Running 2-D Single Precision cases
	bin/spread2d_test_32 1 1 16 16
	bin/spread2d_test_32 2 1 16 16
	bin/spread2d_test_32 1 1 1024 1024
	bin/spread2d_test_32 2 1 1024 1024
	bin/interp2d_test_32 1 1 16 16
	bin/interp2d_test_32 1 1 1024 1024
	bin/cufinufft2d1_test_32 1 8 8
	bin/cufinufft2d1_test_32 2 8 8
	bin/cufinufft2d1_test_32 1 256 256
	bin/cufinufft2d1_test_32 2 512 512
	bin/cufinufft2d2_test_32 1 8 8
	bin/cufinufft2d2_test_32 2 8 8
	bin/cufinufft2d2_test_32 1 256 256
	bin/cufinufft2d2_test_32 2 512 512
	@echo Running 2-D High Density Single Precision cases
	bin/cufinufft2d1_test_32 1 64 64 8192
	bin/cufinufft2d1_test_32 2 64 64 8192
	bin/cufinufft2d2_test_32 1 64 64 8192
	bin/cufinufft2d2_test_32 2 64 64 8192
	@echo Running 2-D Low Density Single Precision cases
	bin/cufinufft2d1_test_32 1 64 64 1024
	bin/cufinufft2d1_test_32 2 64 64 1024
	bin/cufinufft2d2_test_32 1 64 64 1024
	bin/cufinufft2d2_test_32 2 64 64 1024
	@echo Running 2-D-Many Single Precision cases
	bin/cufinufft2d1many_test_32 1 64 64 128 1e-3
	bin/cufinufft2d1many_test_32 1 256 256 1024
	bin/cufinufft2d1many_test_32 2 512 512 256
	bin/cufinufft2d1many_test_32 1 1e2 2e2 3e2 16 1e4
	bin/cufinufft2d1many_test_32 2 1e2 2e2 3e2 16 1e4
	bin/cufinufft2d2many_test_32 1 64 64 128 1e-3
	bin/cufinufft2d2many_test_32 1 256 256 1024
	bin/cufinufft2d2many_test_32 2 512 512 256
	bin/cufinufft2d2many_test_32 1 256 256 1024
	bin/cufinufft2d2many_test_32 1 1e2 2e2 3e2 16 1e4
	bin/cufinufft2d2many_test_32 2 1e2 2e2 3e2 16 1e4

check3D: check3D_32 check3D_64

check3D_64: spreadtest libtest
	@echo Running 3-D Single Precision cases
	# note test method 2 will fail due to shmem limits
	bin/spread3d_test 1 1 16 16 16
	bin/spread3d_test 1 1 512 512 512
	bin/interp3d_test 1 1 16 16 16
	bin/interp3d_test 1 1 512 512 512
	bin/cufinufft3d1_test 1 16 16 16 4096 1e-3
	bin/cufinufft3d1_test 4 15 15 15 2048 1e-3
	bin/cufinufft3d2_test 1 16 16 16 4096 1e-3
	bin/cufinufft3d1_test 1 128 128 128
	bin/cufinufft3d1_test 4 15 15 15
	bin/cufinufft3d2_test 1 16 16 16
	bin/cufinufft3d1_test 1 64 64 64 1000
	bin/cufinufft3d1_test 1 1e2 2e2 3e2 1e4
	bin/cufinufft3d1_test 4 1e2 2e2 3e2 1e4
	bin/cufinufft3d2_test 1 1e2 2e2 3e2

check3D_32: spreadtest libtest
	@echo Running 3-D Single Precision cases
	bin/spread3d_test_32 1 1 16 16 16
	bin/spread3d_test_32 2 1 16 16 16
	bin/spread3d_test_32 1 1 512 512 512
	bin/spread3d_test_32 2 1 512 512 512
	bin/interp3d_test_32 1 1 16 16 16
	bin/interp3d_test_32 1 1 512 512 512
	bin/cufinufft3d1_test_32 1 16 16 16 4096 1e-3
	bin/cufinufft3d1_test_32 2 16 16 16 8192 1e-3
	bin/cufinufft3d1_test_32 4 15 15 15 2048 1e-3
	bin/cufinufft3d2_test_32 1 16 16 16 4096 1e-3
	bin/cufinufft3d2_test_32 2 16 16 16 8192 1e-3
	bin/cufinufft3d1_test_32 1 128 128 128
	bin/cufinufft3d1_test_32 2 16 16 16
	bin/cufinufft3d1_test_32 4 15 15 15
	bin/cufinufft3d2_test_32 1 16 16 16
	bin/cufinufft3d2_test_32 2 16 16 16
	bin/cufinufft3d1_test_32 1 64 64 64 1000
	bin/cufinufft3d1_test_32 2 64 64 64 10000
	bin/cufinufft3d1_test_32 1 1e2 2e2 3e2 1e4
	bin/cufinufft3d1_test_32 2 1e2 2e2 3e2 1e4
	bin/cufinufft3d1_test_32 4 1e2 2e2 3e2 1e4
	bin/cufinufft3d2_test_32 1 1e2 2e2 3e2
	bin/cufinufft3d2_test_32 2 1e2 2e2 3e2

checkexamples: examples
	$(BINDIR)/example2d1many
	$(BINDIR)/example2d2many
# --------------------------------------------- end of check tasks ---------


# Python, some users may want to use pip3 here.
python:
	pip install .

# Docker, for distribution and generation of PyPI wheels.
docker: docker_manylinux2010_x86_64 docker_manylinux2014_x86_64
docker_manylinux2010_x86_64:
	docker build --no-cache -f ci/docker/cuda10.1/Dockerfile-x86_64 -t test_cufinufft_manylinux2010 .
docker_manylinux2014_x86_64:
	docker build --no-cache -f ci/docker/cuda11.0/Dockerfile-x86_64 -t test_cufinufft_manylinux2014 .

wheels: docker_manylinux2010_x86_64
	docker run --gpus all -it -v ${PWD}/wheelhouse:/io/wheelhouse -e PLAT=manylinux2010_x86_64 test_cufinufft_manylinux2010 /io/ci/build-wheels.sh


# Cleanup and phony targets

clean:
	rm -f *.o
	rm -f test/*.o
	rm -f src/*.o
	rm -f src/2d/*.o
	rm -f src/3d/*.o
	rm -f contrib/*.o
	rm -f examples/*.o
	rm -f example2d1
	rm -rf $(BINDIR)
	rm -rf lib
	rm -rf lib-static

.PHONY: default all libtest spreadtest check checkspread checkapi checkexamples
.PHONY: check2D check2D_32 check2D_64
.PHONY: check3D check3D_32 check3D_64
.PHONY: python docker docker_manylinux2010_x86_64 docker_manylinux2014_x86_64
.PHONY: wheels clean
