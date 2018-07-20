CXX=g++
NVCC=nvcc
CXXFLAGS=-DTIME
#CXXFLAGS=-DINFO -DDEBUG -DRESULT -DTIME
NVCCFLAGS=-arch=sm_50
INC=-I/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/samples/common/inc/ \
    -I/mnt/home/yshih/cub/

spread1d: spread1d.o utils.o main_1d.cu
	$(NVCC) main_1d.cu -o spread1d spread1d.o utils.o $(INC)

spread1d.o: utils.o spread1d.cu
	$(NVCC) -c spread1d.cu $(INC)

main_2d.o: examples/main_2d.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS) $(NVCCFLAGS) $(INC)

spread2d_wrapper.o: src/spread2d_wrapper.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS) $(NVCCFLAGS) $(INC)

spread2d.o: src/spread2d.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS) $(NVCCFLAGS) $(INC)

utils.o: src/finufft/utils.cpp
	$(CC) -c $< -o $@ $(CXXFLAGS)

cnufftspread.o: src/finufft/cnufftspread.cpp
	$(CC) -c $< -o $@ $(CXXFLAGS) $(INC)

accuracycheck_2d.o: test/accuracycheck_2d.cu
	$(NVCC) -c $< $(CXXFLAGS) $(NVCCFLAGS) -o $@ $(INC)

spread2d: main_2d.o spread2d_wrapper.o spread2d.o utils.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

accuracy: accuracycheck_2d.o spread2d_wrapper.o spread2d.o utils.o cnufftspread.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm *.o
	rm -f spread1d
	rm -f spread2d
	rm -f accuracy

