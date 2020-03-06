CC=gcc
CXX=g++
NVCC=nvcc

CXXFLAGS= -DNEED_EXTERN_C  -fPIC -O3 -funroll-loops -march=native -g -std=c++11
#NVCCFLAGS=-DINFO -DDEBUG -DRESULT -DTIME
NVCCFLAGS= -std=c++11 -ccbin=$(CXX) -O3 -DTIME -arch=sm_70 \
	--default-stream per-thread -Xcompiler "$(CXXFLAGS)"
#If using any card with architecture KXX, change to -arch=sm_30 (see GPUs 
#supported section in https://en.wikipedia.org/wiki/CUDA for more info)
#DEBUG add "-g -G" for cuda-gdb debugger

ifeq ($(PREC),SINGLE)
PRECSUFFIX=f
CXXFLAGS+=-DSINGLE
NVCCFLAGS+=-DSINGLE
else
PRECSUFFIX=
endif

INC=-I/cm/shared/sw/pkg/devel/cuda/9.0.176/samples/common/inc/ \
    -I/mnt/home/yshih/cub/ \
    -I/cm/shared/sw/pkg/devel/cuda/9.0.176/include/
LIBS_PATH=

FFTWNAME=fftw3
FFTW=$(FFTWNAME)$(PRECSUFFIX)

LIBS=-lm -lcudart -lstdc++ -lnvToolsExt -lcufft -lcuda -l$(FFTW)

LIBNAME=libcufinufft$(PRECSUFFIX)
DYNAMICLIB=lib/$(LIBNAME).so
STATICLIB=lib-static/$(LIBNAME).a

CLIBNAME=libcufinufftc$(PRECSUFFIX)
DYNAMICCLIB=lib/$(CLIBNAME).so

HEADERS = src/cufinufft.h src/deconvolve.h src/memtrasfer.h src/profile.h \
	src/spreadinterp.h
FINUFFTOBJS=finufft/utils.o finufft/dirft2d.o finufft/common.o \
	finufft/spreadinterp.o finufft/contrib/legendre_rule_fast.o

CUFINUFFTOBJS=src/2d/spreadinterp2d.o src/2d/cufinufft2d.o \
	src/2d/spread2d_wrapper.o src/2d/spread2d_wrapper_paul.o \
	src/2d/interp2d_wrapper.o src/memtransfer_wrapper.o \
	src/deconvolve_wrapper.o src/cufinufft.o src/profile.o \
	src/3d/spreadinterp3d.o src/3d/spread3d_wrapper.o \
	src/3d/interp3d_wrapper.o src/3d/cufinufft3d.o
	
CUFINUFFTCOBJS=src/cufinufftc.o
#-include make.inc

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.c
	$(CC) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INC) $< -o $@


spread2d: test/spread_2d.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

interp2d: test/interp_2d.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

spreadinterp_test: test/spreadinterp_test.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

finufft2d_test: test/finufft2d_test.o finufft/finufft2d.o $(CUFINUFFTOBJS) \
	$(FINUFFTOBJS)
	$(CXX) $^ $(LIBS_PATH) $(LIBS) -o $@

cufinufft_test: test/cufinufft_test.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) -o $@

cufinufft2d1_test: test/cufinufft2d1_test.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) -o $@

cufinufft2d1many_test: test/cufinufft2d1many_test.o $(CUFINUFFTOBJS) \
	$(FINUFFTOBJS)
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) -o $@

cufinufft2d2_test: test/cufinufft2d2_test.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) -o $@

cufinufft2d2many_test: test/cufinufft2d2many_test.o $(CUFINUFFTOBJS) \
	$(FINUFFTOBJS)
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) -o $@

spread3d: test/spread_3d.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

interp3d: test/interp_3d.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

spreadinterp3d_test: test/spreadinterp3d_test.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $(NVCCFLAGS) $(LIBS) -o $@ $^

cufinufft3d1_test: test/cufinufft3d1_test.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) $(LIBS_CUFINUFFT) -o $@

cufinufft3d2_test: test/cufinufft3d2_test.o $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) $(LIBS_CUFINUFFT) -o $@

lib: $(STATICLIB) $(DYNAMICLIB)

clib: $(DYNAMICCLIB)

$(STATICLIB): $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	mkdir -p lib-static
	ar rcs $(STATICLIB) $(CUFINUFFTOBJS) $(FINUFFTOBJS)
$(DYNAMICLIB): $(CUFINUFFTOBJS) $(FINUFFTOBJS)
	mkdir -p lib	
	$(NVCC) -shared $(NVCCFLAGS) $(CUFINUFFTOBJS) $(FINUFFTOBJS) -o $(DYNAMICLIB) $(LIBS) 

$(DYNAMICCLIB): $(CUFINUFFTCOBJS) $(STATICLIB)
	mkdir -p lib
	gcc -shared -o $(DYNAMICCLIB) $(CUFINUFFTCOBJS) $(STATICLIB) $(LIBS)

all: spread2d interp2d spreadinterp_test finufft2d_test cufinufft2d1_test \
	cufinufft2d2_test cufinufft2d1many_test cufinufft2d2many_test spread3d \
	interp3d cufinufft3d1_test cufinufft3d2_test spreadinterp3d_test \
	lib


clean:
	rm -f *.o
	rm -f test/*.o
	rm -f src/*.o
	rm -f src/2d/*.o
	rm -f src/3d/*.o
	rm -f finufft/*.o
	rm -f finufft/contrib/*.o
	rm -f examples/*.o
	rm -f spread2d
	rm -f accuracy
	rm -f interp2d
	rm -f finufft2d_test
	rm -f cufinufft2d1_test
	rm -f cufinufft2d2_test
	rm -f cufinufft2d1many_test
	rm -f cufinufft2d2many_test
	rm -f spread3d
	rm -f interp3d
	rm -f cufinufft3d1_test
	rm -f cufinufft3d2_test
	rm -f spreadinterp_test
	rm -f spreadinterp3d_test
	rm -f example2d1
	rm -f lib/*.so
	rm -f lib-static/*.a
	rmdir lib lib-static
