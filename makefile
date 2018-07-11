CC=g++
NVCC=nvcc
INCLUDE_DIR=/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/samples/common/inc/
spread1d: spread1d.o utils.o main_1d.cu
	$(NVCC) main_1d.cu -o spread1d spread1d.o utils.o -I$(INCLUDE_DIR)
spread1d.o: utils.o spread1d.cu
	$(NVCC) -c spread1d.cu -I$(INCLUDE_DIR)

main_2d.o: main_2d.cu
	$(NVCC) main_2d.cu -c main_2d.o -I$(INCLUDE_DIR) --ptxas-options=-v
spread2d_wrapper.o: spread2d_wrapper.cu
	$(NVCC) -c spread2d_wrapper.cu -I$(INCLUDE_DIR) --ptxas-options=-v
spread2d.o: spread2d.cu
	$(NVCC) -c spread2d.cu -I$(INCLUDE_DIR) --ptxas-options=-v
utils.o: utils.cpp
	$(CC) -c utils.cpp 
scan.o:scan.cu
	$(NVCC) -I$(INCLUDE_DIR) -o $@ -c $< --ptxas-options=-v

spread2d: main_2d.o spread2d_wrapper.o spread2d.o utils.o scan.o
	$(NVCC) -o spread2d main_2d.o spread2d_wrapper.o spread2d.o utils.o scan.o --ptxas-options=-v

clean:
	rm *.o
	rm -f spread1d
	rm -r spread2d

