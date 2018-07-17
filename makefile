CC=g++
NVCC=nvcc
CXXFLAGS=-DTIME
#CXXFLAGS=-DINFO -DDEBUG -DRESULT -DTIME
INCLUDE_DIR=/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/samples/common/inc/

spread1d: spread1d.o utils.o main_1d.cu
	$(NVCC) main_1d.cu -o spread1d spread1d.o utils.o -I$(INCLUDE_DIR)
spread1d.o: utils.o spread1d.cu
	$(NVCC) -c spread1d.cu -I$(INCLUDE_DIR)

main_2d.o: main_2d.cu
	$(NVCC) main_2d.cu -c $(CXXFLAGS) main_2d.o -I$(INCLUDE_DIR)
spread2d_wrapper.o: spread2d_wrapper.cu
	$(NVCC) -c $(CXXFLAGS) spread2d_wrapper.cu -I$(INCLUDE_DIR)
spread2d.o: spread2d.cu
	$(NVCC) -c $(CXXFLAGS) spread2d.cu -I$(INCLUDE_DIR)
utils.o: utils.cpp
	$(CC) -c $(CXXFLAGS) utils.cpp 

spread2d: main_2d.o spread2d_wrapper.o spread2d.o utils.o
	$(NVCC) -o spread2d main_2d.o spread2d_wrapper.o spread2d.o utils.o

clean:
	rm *.o
	rm -f spread1d
	rm -f spread2d

