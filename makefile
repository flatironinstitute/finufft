CC=nvcc
INCLUDE_DIR=/mnt/xfs1/flatiron-sw/pkg/devel/cuda/8.0.61/samples/common/inc/
spread1d: spread1d.o main.cu
	$(CC) main.cu -o spread1d spread1d.o -I$(INCLUDE_DIR)
spread1d.o: spread1d.cu
	$(CC) -c spread1d.cu -I$(INCLUDE_DIR)
spread2d: spread2d.o main2.cu
	$(CC) main2.cu -o spread2d spread2d.o -I$(INCLUDE_DIR)
spread2d.o: spread2d.cu
	$(CC) -c spread2d.cu -I$(INCLUDE_DIR)
clean:
	rm *.o
	rm -f spread1d
	rm -r spread2d

