# makefile overrides to use Intel ICC compiler & libiomp5. double prec only.
# (fixed the fatal mistake of leaving -lgomp in the compile line! 6/4/20)

CXX=icpc
CC=icc
FC=ifort

# we want to start fresh, ignore the GCC flags...
CFLAGS = -O3 -xHost
# CFLAGS += -lsvml    % fails (gives nans & no faster)

CXXFLAGS = $(CFLAGS)
FFLAGS   = $(CFLAGS)

# It is crucial to *replace* -fopenmp & -lgomp with Intel's equiv
# (crucial in the sense that linking libiomp5 *and* libgomp gives weird
# segfaults, corruption in parallel blocks, or works fine, frustratingly):
OMPFLAGS = -qopenmp
OMPLIBS =
