CC=gcc
CXX=g++
NVCC=nvcc
CXXFLAGS=-DNEED_EXTERN_C -fPIC -Ofast -funroll-loops -march=native -g
#NVCCFLAGS=-DINFO -DDEBUG -DRESULT -DTIME
NVCCFLAGS=-arch=sm_50 #If using any card with architecture KXX, change to -arch=sm_30 (see GPUs supported section in https://en.wikipedia.org/wiki/CUDA for more info)
INC=-I/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/samples/common/inc/ \
    -I/mnt/home/yshih/cub/ \
    -I/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/include/
LIBS_PATH=
LIBS=-lm -lfftw3 -lcudart -lstdc++
LIBS_CUFINUFFT=-lcufft

-include make.inc

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INC) $< -o $@

spread1d: examples/spread_1d.o src/1d/spread1d_wrapper.o src/1d/spread1d.o finufft/utils.o src/memtransfer_wrapper.o\
          src/common.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

spread2d: examples/spread_2d.o src/2d/spread2d_wrapper.o src/2d/spread2d.o finufft/utils.o src/memtransfer_wrapper.o\
          src/common.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

interp2d: examples/interp_2d.o src/2d/spread2d_wrapper.o src/2d/spread2d.o src/2d/interp2d_wrapper.o src/2d/interp2d.o \
          finufft/utils.o src/memtransfer_wrapper.o src/common.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

spreadinterp_test: test/spreadinterp_test.o src/2d/spread2d_wrapper.o src/2d/spread2d.o finufft/utils.o \
                   finufft/spreadinterp.o src/memtransfer_wrapper.o src/2d/interp2d_wrapper.o src/2d/interp2d.o \
                   src/common.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

finufft2d_test: test/finufft2d_test.o finufft/finufft2d.o finufft/utils.o finufft/spreadinterp.o \
                finufft/dirft2d.o finufft/common.o \
                finufft/contrib/legendre_rule_fast.o src/2d/spread2d_wrapper.o src/2d/spread2d.o \
                src/2d/cufinufft2d.o src/deconvolve_wrapper.o src/memtransfer_wrapper.o \
                src/2d/interp2d_wrapper.o src/2d/interp2d.o
	$(CXX) $^ $(LIBS_PATH) $(LIBS) $(LIBS_CUFINUFFT) -o $@

cufinufft2d1_test: examples/cufinufft2d1_test.o finufft/utils.o finufft/dirft2d.o finufft/common.o \
                   finufft/spreadinterp.o finufft/contrib/legendre_rule_fast.o src/2d/spread2d_wrapper.o src/2d/spread2d.o \
                   src/2d/cufinufft2d.o src/deconvolve_wrapper.o src/memtransfer_wrapper.o src/2d/interp2d_wrapper.o src/2d/interp2d.o
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) $(LIBS_CUFINUFFT) -o $@

cufinufft2d1many_test: examples/cufinufft2d1many_test.o finufft/utils.o finufft/dirft2d.o finufft/common.o \
	               finufft/spreadinterp.o finufft/contrib/legendre_rule_fast.o src/2d/spread2d_wrapper.o src/2d/spread2d.o \
                       src/2d/cufinufft2d.o src/deconvolve_wrapper.o src/memtransfer_wrapper.o src/2d/interp2d_wrapper.o src/2d/interp2d.o
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) $(LIBS_CUFINUFFT) -o $@

cufinufft2d2_test: examples/cufinufft2d2_test.o finufft/utils.o finufft/dirft2d.o finufft/common.o \
                   finufft/spreadinterp.o finufft/contrib/legendre_rule_fast.o src/2d/spread2d_wrapper.o src/2d/spread2d.o \
                   src/2d/cufinufft2d.o src/deconvolve_wrapper.o src/memtransfer_wrapper.o src/2d/interp2d_wrapper.o src/2d/interp2d.o
	$(NVCC) $^ $(NVCCFLAGS) $(LIBS_PATH) $(LIBS) $(LIBS_CUFINUFFT) -o $@

all: spread1d spread2d interp2d spreadinterp_test finufft2d_test cufinufft2d1_test cufinufft2d2_test
clean:
	rm -f *.o
	rm -f examples/*.o
	rm -f src/*.o
	rm -f src/2d/*.o
	rm -f finufft/*.o
	rm -f finufft/contrib/*.o
	rm -f test/*.o
	rm -f spread2d
	rm -f accuracy
	rm -f interp2d
	rm -f finufft2d_test
	rm -f cufinufft2d1_test
	rm -f cufinufft2d2_test
