CXX=g++
NVCC=nvcc
CXXFLAGS=-DTIME -DRESULT -DINFO
#CXXFLAGS=-DINFO -DDEBUG -DRESULT -DTIME
NVCCFLAGS=-arch=sm_50
INC=-I/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/samples/common/inc/ \
    -I/mnt/home/yshih/cub/

spread1d: spread1d.o utils.o main_1d.cu
	$(NVCC) main_1d.cu -o spread1d spread1d.o utils.o $(INC)
spread1d.o: utils.o spread1d.cu
	$(NVCC) -c spread1d.cu $(INC)

main_2d.o: main_2d.cu
	$(NVCC) main_2d.cu -c $(CXXFLAGS) $(NVCCFLAGS) main_2d.o $(INC)
spread2d_wrapper.o: spread2d_wrapper.cu
	$(NVCC) -c $(CXXFLAGS) $(NVCCFLAGS) spread2d_wrapper.cu $(INC)
spread2d.o: spread2d.cu
	$(NVCC) -c $(CXXFLAGS) $(NVCCFLAGS) spread2d.cu $(INC)
utils.o: utils.cpp
	$(CC) -c $(CXXFLAGS) utils.cpp 

spread2d: main_2d.o spread2d_wrapper.o spread2d.o utils.o
	$(NVCC) $(NVCCFLAGS) -o spread2d main_2d.o spread2d_wrapper.o spread2d.o utils.o

clean:
	rm *.o
	rm -f spread1d
	rm -f spread2d

