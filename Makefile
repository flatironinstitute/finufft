CC=gcc
CXX=g++
NVCC=nvcc

# Developer-users are suggested to change this in their make.inc, see:
#   http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
#NVARCH = -arch=sm_70
CXXFLAGS= -fPIC -O3 -funroll-loops -march=native -g -std=c++11
NVCCFLAGS= -std=c++11 -ccbin=$(CXX) -O3 $(NVARCH) \
	--default-stream per-thread -Xcompiler "$(CXXFLAGS)"
#DEBUG add "-g -G" for cuda-gdb debugger
#NVCCFLAGS=-DINFO -DDEBUG -DRESULT -DTIME

# CUDA Related build dependencies
CUDA_ROOT=/usr/local/cuda
INC=-I$(CUDA_ROOT)/include \
	-Icontrib/cuda_samples
NVCC_LIBS_PATH=-L$(CUDA_ROOT)/lib64

FFTWNAME=fftw3

LIBS=-lm -lcudart -lstdc++ -lnvToolsExt -lcufft -lcuda -l$(FFTWNAME) -l$(FFTWNAME)f


#############################################################
# Allow the user to override any variable above this point. #
-include make.inc

# Include header files
INC += -I include


LIBNAME=libcufinufft
DYNAMICLIB=lib/$(LIBNAME).so
STATICLIB=lib-static/$(LIBNAME).a

CLIBNAME=libcufinufftc
DYNAMICCLIB=lib/$(CLIBNAME).so

BINDIR=./bin

HEADERS = include/cufinufft.h src/cudeconvolve.h src/memtransfer.h include/profile.h \
	src/cuspreadinterp.h
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

CUFINUFFTCOBJS_64=src/cufinufftc.o
CUFINUFFTCOBJS_32=$(CUFINUFFTCOBJS_64:%.o=%_32.o)

%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.c $(HEADERS)
	$(CC) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.cu $(HEADERS)
	$(NVCC) --device-c -c $(NVCCFLAGS) $(INC) $< -o $@
%_32.o: %.cpp $(HEADERS)
	$(CXX) -DSINGLE -c $(CXXFLAGS) $(INC) $< -o $@
%_32.o: %.c $(HEADERS)
	$(CC) -DSINGLE -c $(CXXFLAGS) $(INC) $< -o $@
%_32.o: %.cu $(HEADERS)
	$(NVCC) -DSINGLE --device-c -c $(NVCCFLAGS) $(INC) $< -o $@


all: $(BINDIR)/spread2d \
	$(BINDIR)/interp2d \
	$(BINDIR)/cufinufft2d1_test \
	$(BINDIR)/cufinufft2d2_test \
	$(BINDIR)/cufinufft2d1many_test \
	$(BINDIR)/cufinufft2d2many_test \
	$(BINDIR)/cufinufft2d1_test_32 \
	$(BINDIR)/cufinufft2d2_test_32 \
	$(BINDIR)/cufinufft2d1many_test_32 \
	$(BINDIR)/cufinufft2d2many_test_32 \
	$(BINDIR)/spread3d \
	$(BINDIR)/interp3d \
	$(BINDIR)/cufinufft3d1_test \
	$(BINDIR)/cufinufft3d2_test \
	$(BINDIR)/cufinufft3d1_test_32 \
	$(BINDIR)/cufinufft3d2_test_32 \
	lib clib

$(BINDIR)/spread2d: test/spread_2d.o $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

$(BINDIR)/interp2d: test/interp_2d.o $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

$(BINDIR)/spread3d: test/spread_3d.o $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

$(BINDIR)/interp3d: test/interp_3d.o $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

$(BINDIR)/%: test/%.o $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@

$(BINDIR)/%_32: test/%_32.o $(CUFINUFFTOBJS_32) $(CUFINUFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) -DSINGLE $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@


lib: $(STATICLIB) $(DYNAMICLIB)

clib: $(DYNAMICCLIB)

$(STATICLIB): $(CUFINUFFTOBJS) $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS_32) $(CONTRIBOBJS)
	mkdir -p lib-static
	ar rcs $(STATICLIB) $^
$(DYNAMICLIB): $(CUFINUFFTOBJS) $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS_32) $(CONTRIBOBJS)
	mkdir -p lib
	$(NVCC) -shared $(NVCCFLAGS) $^ -o $(DYNAMICLIB) $(LIBS)

$(DYNAMICCLIB): $(CUFINUFFTOBJS) $(CUFINUFFTCOBJS_64) $(CUFINUFFTCOBJS_32) $(STATICLIB)
	mkdir -p lib
	gcc -shared -o $(DYNAMICCLIB) $^ $(NVCC_LIBS_PATH) $(LIBS)


# Check targets

check: all
	$(MAKE) check2D
	$(MAKE) check3D

check2D: all check2D_64 check2D_32

check2D_64:
	@echo Running 2-D cases
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

check2D_32:
	@echo Running 2-D Single Precision cases
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


check3D: all check3D_32 check3D_64

check3D_64: all
	@echo Running 3-D Single Precision cases
	# note test method 2 will fail due to shmem limits
	bin/cufinufft3d1_test 1 16 16 16 4096 1e-3
	bin/cufinufft3d1_test 4 15 15 15 2048 1e-3
	bin/cufinufft3d2_test 1 16 16 16 4096 1e-3
	bin/cufinufft3d1_test 1 128 128 128
	bin/cufinufft3d1_test 2 16 16 16
	bin/cufinufft3d1_test 4 15 15 15
	bin/cufinufft3d2_test 1 16 16 16
	bin/cufinufft3d1_test 1 64 64 64 1000
	bin/cufinufft3d1_test 1 1e2 2e2 3e2 1e4
	bin/cufinufft3d1_test 4 1e2 2e2 3e2 1e4
	bin/cufinufft3d2_test 1 1e2 2e2 3e2

check3D_32: all
	@echo Running 3-D Single Precision cases
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


.PHONY: all
.PHONY: check
.PHONY: check2D
.PHONY: check3D
.PHONY: clean
.PHONY: clib
.PHONY: lib
