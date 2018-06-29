CC=nvcc
INCLUDE_DIR=/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/samples/common/inc/
spread1d: spread1d.o
	$(CC) main.cu -o spread1d spread1d.o -I$(INCLUDE_DIR)
spread1d.o: spread1d.cu
	$(CC) -c spread1d.cu -I$(INCLUDE_DIR)
clean:
	rm *.o
	rm spread1d
