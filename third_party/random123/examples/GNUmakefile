no_target_specified: runcore
	@echo
	@echo The default make rule is equivalent to \'make runcore\' which builds
	@echo and runs most of the examples.
	@echo The following \'meta-targets\' are available:
	@echo "      " $(meta_targets)
	@echo Here is the complete list of individual program targets:
	@echo "      " $(all_primary_targets)
	@echo Prepend \'run\' to any of the program targets or metatargets
	@echo to run the binary and check for a zero exit status.
	@echo Adding force=1 on the command line causes all targets to be considered out-of-date.
.PHONY: no_target_specified

# metatargets are variables which get mapped by METATARGET_template
meta_targets:=core aesni c cpp gsl cuda opencl thread metal

# Platform metatargets: each one typically has specific requirements in the build environment.
# c is C99 (will work in MSVC), cpp is C++98, gsl requires the GNU Scientific Library
# (specifically, the gsl-config program in the PATH), thread requires POSIX threads,
# CUDA requires NVIDIA CUDA 3.x or newer, OpenCL requires OpenCL includes & libraries
# (e.g. AMD APP SDK, NVIDIA SDK)
c:=simple pi_capi
cpp:=simplepp pi_uniform pi_cppapi pi_microurng
gsl:=pi_gsl
cuda:=pi_cuda pi_cudapp
opencl:=pi_opencl
thread:=  # ?? should we have a thread example?
metal:=pi_metal

# Convenience metatargets: these are to help developers test functional subsets across platforms
core:=$(c) $(cpp)
aesni:=pi_aes

$(gsl) : override LDLIBS += `gsl-config --libs`
$(gsl) : override CFLAGS += `gsl-config --cflags`

$(opencl) : % : %_kernel.i
# Or try this if we USE_GENCL in kat_opencl.c
#$(opencl) : override CPPFLAGS += -DSRCDIR=\"$(dir $(abspath $<)).\"

ifeq ($(shell uname),Darwin)
$(opencl) : override LDLIBS+=-framework OpenCL
else
$(opencl) : override LDLIBS+=-lOpenCL
endif
$(opencl) : override CFLAGS+=-I.
# Note, the Intel OpenCL SDK (1.5) has unresolved C++ symbols in its
# libOpenCL.so Even though 'main' is a C program, you may need to link
# it with a C++ compiler-driver, e.g., g++.  Since this Makefile does
# compile-and-link in one step, use something like:
# $(opencl) : CC=g++ -xc
# which will invoke the g++ compiler-driver, but will treat the
# program as C rather than C++.

$(metal) : % : %_kernel.metallib
$(metal) : override LDLIBS+=-framework Metal -framework Foundation -framework CoreGraphics

all_primary_targets += $(addsuffix _kernel.i, $(opencl))
all_primary_targets += $(addsuffix _kernel.metallib, $(metal))
all_primary_targets += $(addsuffix _kernel.air, $(metal))

################################################
# Generic boilerplate from here down:
vpath %.c $(srcdir/)
vpath %.cpp $(srcdir/)
vpath %.cu $(srcdir/)
vpath %.ocl $(srcdir/)
vpath %.metal $(srcdir/)

define METATARGET_template
.PHONY: $(1)
$(1) : $(filter-out $(SKIP_TARGETS), $($(1)))
.PHONY: run$(1)
run$(1) : $(addprefix run, $(filter-out $(SKIP_TARGETS), $($(1))))
all_primary_targets += $($(1))
endef

$(foreach T,$(meta_targets), $(eval $(call METATARGET_template,$(T))))

# sort also does 'uniq'
all_primary_targets:=$(sort $(all_primary_targets))

INC=$(srcdir/)../include
override CPPFLAGS += -I$(INC)

ifndef NVCC
NVCC:=nvcc
endif
# The rngs are *very* slow without optimization.  In the simplest case,
# where the user just calls 'make', we don't want them to see terrible
# performance.  Unfortunately, this might surprise someone
# who says, e.g., make CPPFLAGS=-O0.  Oh well...
ifndef CFLAGS
CFLAGS:=-O
endif
ifndef CXXFLAGS
CXXFLAGS:=-O
endif

%.i : %.ocl
	CC="$(CC)" CPPFLAGS="$(CPPFLAGS)" $(srcdir/)./gencl.sh $< > $@

% : %.cu
	$(NVCC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH) $< $(LOADLIBES) $(LDLIBS) -o $@

% : %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH) $< $(LOADLIBES) $(LDLIBS) -o $@

% : %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH) $< $(LOADLIBES) $(LDLIBS) -o $@

% : %.m
	$(CC) $(OBJCFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH) $< $(LOADLIBES) $(LDLIBS) -o $@

%.air : %.metal
	xcrun --sdk macosx metal $(CPPFLAGS) -c $< -o $@

%.metallib : %.air
	xcrun --sdk macosx metallib $< -o $@

run% : %
	./$^ $(RUN_ARGS)

# In lieu of autodepends, just say that all the compilation targets depend on all the headers.
hdrs:=$(wildcard $(srcdir/)*.h $(srcdir/)gsl/*.c $(INC)/Random123/*.h $(INC)/Random123/*.hpp $(INC)/Random123/*/*.h $(INC)/Random123/*/*.hpp)
misc:=$(wildcard $(srcdir/)*.cu $(srcdir/)*.ocl)
$(all_primary_targets) : $(hdrs)
$(misc) : $(hdrs)

# If you put force=y on the command line, then $(all_primary_targets) will be
# depend on FORCE, and hence will not be up-to-date.
ifdef force
$(all_primary_targets) : FORCE
FORCE:
endif

.PHONY : echo_build_commands
echo_build_commands:
	make -n force=1 $(all_primary_targets) | grep -v 'is up to date'

.PHONY : clean veryclean
clean:
	rm -f $(all_primary_targets)

veryclean:
	rm -f $(all_primary_targets) *.o \#* *~ *.pdb *.exe *.obj *.ilk *.suo

.PHONY : install

# N.B. normally these are exported by ../GNUmakefile
prefix?=/usr/local
datarootdir?=$(prefix)/share
docdir?=$(datarootdir)/doc/Random123
install:
	mkdir -p $(DESTDIR)$(docdir)/examples
	cp README GNUmakefile *.sh *.c *.h  *.cpp *.cu *.metal *.m *.ocl $(DESTDIR)$(docdir)/examples
