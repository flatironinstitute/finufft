CC=gcc
CXX=g++
NVCC=nvcc
CXXFLAGS=-DNEED_EXTERN_C -fPIC -Ofast -funroll-loops -march=native
#CXXFLAGS=-DINFO -DDEBUG -DRESULT -DTIME
NVCCFLAGS=-arch=sm_50 -DTIME
INC=-I/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/samples/common/inc/ \
    -I/mnt/home/yshih/cub/
LIBS_PATH=
LIBS=-lm -lfftw3 -lcudart -lstdc++

-include make.inc

spread1d: spread1d.o utils.o main_1d.cu
	$(NVCC) main_1d.cu -o spread1d spread1d.o utils.o $(INC)

spread1d.o: utils.o spread1d.cu
	$(NVCC) -c spread1d.cu $(INC)

main_2d.o: examples/main_2d.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INC)

spread2d_wrapper.o: src/spread2d_wrapper.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INC)

spread2d.o: src/spread2d.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INC)

utils.o: src/finufft/utils.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INC)

cnufftspread.o: src/finufft/cnufftspread.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INC)

common.o: src/finufft/common.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INC)

dirft2d.o: src/finufft/dirft2d.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INC)

finufft2d.o: src/finufft/finufft2d.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INC)

legendre_rule_fast.o: src/finufft/contrib/legendre_rule_fast.c
	$(CC) -c $< -o $@ $(INC)

finufft2d_test.o: test/finufft2d_test.cpp
	$(CXX) -c $< $(CXXFLAGS) $(INC) -o $@

accuracycheck_2d.o: test/accuracycheck_2d.cu
	$(NVCC) -c $< $(NVCCFLAGS) -o $@ $(INC)

spread2d: main_2d.o spread2d_wrapper.o spread2d.o utils.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

accuracy: accuracycheck_2d.o spread2d_wrapper.o spread2d.o utils.o cnufftspread.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

finufft2d_test: finufft2d_test.o finufft2d.o utils.o cnufftspread.o dirft2d.o common.o legendre_rule_fast.o spread2d_wrapper.o spread2d.o
	$(CXX) $^ $(LIBS_PATH) $(LIBS) -o $@

clean:
	rm *.o
	rm -f spread1d
	rm -f spread2d
	rm -f accuracy
	rm -f finufft2d_test

