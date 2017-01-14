# Makefile for CNUFFT

CC=g++
CFLAGS=-fPIC -O2
FC=gfortran
FFLAGS=-fPIC -O2

OBJS = besseli.o cnufftspread.o cnufftspread_c.o

default: spreadtest

# generic libs
.cpp.o:
	$(CC) -c $(CFLAGS) $<

.f.o:
	$(CC) -c $(CFLAGS) $<

spreadtest: $(OBJS)
	$(CC) $(CFLAGS) spreadtest.cpp $(OBJS) -o spreadtest
