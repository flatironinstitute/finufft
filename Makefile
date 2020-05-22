CC=gcc
CXX=g++
NVCC=nvcc

# We'll sacrifice longer compile times for broader compatibility out of the box.
# Developer-users are suggested to change this in their make.inc, see:
#   http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
NVARCH = -arch=sm_70 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_75,code=compute_75 

CXXFLAGS= -DNEED_EXTERN_C  -fPIC -O3 -funroll-loops -march=native -g -std=c++11
#NVCCFLAGS=-DINFO -DDEBUG -DRESULT -DTIME
NVCCFLAGS= -std=c++11 -ccbin=$(CXX) -O3 -DTIME $(NVARCH) \
	--default-stream per-thread -Xcompiler "$(CXXFLAGS)"
#DEBUG add "-g -G" for cuda-gdb debugger

# CUDA Related build dependencies
CUDA_ROOT=/usr/local/cuda
CUB_ROOT=./cub
INC=-I$(CUDA_ROOT)/include \
	-Icontrib/cuda_samples \
	-I$(CUB_ROOT)
NVCC_LIBS_PATH=-L$(CUDA_ROOT)/lib64

FFTWNAME=fftw3
FFTW=$(FFTWNAME)$(PRECSUFFIX)

LIBS=-lm -lcudart -lstdc++ -lnvToolsExt -lcufft -lcuda -l$(FFTW)


#############################################################
# Allow the user to override any variable above this point. #
-include make.inc

# Include header files
INC += -I include

ifeq ($(PREC),SINGLE)
PRECSUFFIX=f
CXXFLAGS+=-DSINGLE
NVCCFLAGS+=-DSINGLE
else
PRECSUFFIX=
endif

LIBNAME=libcufinufft$(PRECSUFFIX)
DYNAMICLIB=lib/$(LIBNAME).so
STATICLIB=lib-static/$(LIBNAME).a

CLIBNAME=libcufinufftc$(PRECSUFFIX)
DYNAMICCLIB=lib/$(CLIBNAME).so

BINDIR=./bin

HEADERS = src/cufinufft.h src/deconvolve.h src/memtrasfer.h src/profile.h \
	src/spreadinterp.h
CONTRIBOBJS=contrib/utils.o contrib/dirft2d.o contrib/common.o \
	contrib/spreadinterp.o contrib/legendre_rule_fast.o

CUFINUFFTOBJS=src/2d/spreadinterp2d.o src/2d/cufinufft2d.o \
	src/2d/spread2d_wrapper.o src/2d/spread2d_wrapper_paul.o \
	src/2d/interp2d_wrapper.o src/memtransfer_wrapper.o \
	src/deconvolve_wrapper.o src/cufinufft.o src/profile.o \
	src/3d/spreadinterp3d.o src/3d/spread3d_wrapper.o \
	src/3d/interp3d_wrapper.o src/3d/cufinufft3d.o

CUFINUFFTCOBJS=src/cufinufftc.o

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.c
	$(CC) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INC) $< -o $@


$(BINDIR)/spread2d: test/spread_2d.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

$(BINDIR)/interp2d: test/interp_2d.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

$(BINDIR)/cufinufft_test: test/cufinufft_test.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@

$(BINDIR)/cufinufft2d1_test: test/cufinufft2d1_test.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@

$(BINDIR)/cufinufft2d1many_test: test/cufinufft2d1many_test.o $(CUFINUFFTOBJS) \
	$(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@

$(BINDIR)/cufinufft2d2_test: test/cufinufft2d2_test.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@

$(BINDIR)/cufinufft2d2many_test: test/cufinufft2d2many_test.o $(CUFINUFFTOBJS) \
	$(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@

$(BINDIR)/spread3d: test/spread_3d.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

$(BINDIR)/interp3d: test/interp_3d.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

$(BINDIR)/cufinufft3d1_test: test/cufinufft3d1_test.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) $(LIBS_CUFINUFFT) -o $@

$(BINDIR)/cufinufft3d2_test: test/cufinufft3d2_test.o $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) $(LIBS_CUFINUFFT) -o $@

lib: $(STATICLIB) $(DYNAMICLIB)

clib: $(DYNAMICCLIB)

$(STATICLIB): $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p lib-static
	ar rcs $(STATICLIB) $(CUFINUFFTOBJS) $(CONTRIBOBJS)
$(DYNAMICLIB): $(CUFINUFFTOBJS) $(CONTRIBOBJS)
	mkdir -p lib
	$(NVCC) -shared $(NVCCFLAGS) $(CUFINUFFTOBJS) $(CONTRIBOBJS) -o $(DYNAMICLIB) $(LIBS)

$(DYNAMICCLIB): $(CUFINUFFTCOBJS) $(STATICLIB)
	mkdir -p lib
	gcc -shared -o $(DYNAMICCLIB) $(CUFINUFFTCOBJS) $(STATICLIB) $(NVCC_LIBS_PATH) $(LIBS)

all: $(BINDIR)/spread2d \
	$(BINDIR)/interp2d \
	$(BINDIR)/cufinufft2d1_test \
	$(BINDIR)/cufinufft2d2_test \
	$(BINDIR)/cufinufft2d1many_test \
	$(BINDIR)/cufinufft2d2many_test \
	$(BINDIR)/spread3d \
	$(BINDIR)/interp3d \
	$(BINDIR)/cufinufft3d1_test \
	$(BINDIR)/cufinufft3d2_test \
	lib clib


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
