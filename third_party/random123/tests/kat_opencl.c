/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "util_opencl.h"
#include "kat_main.h"

// USE_GENCL:  gencl.sh is a small shell script
// that pre-processes foo.ocl into foo.i, containing
// a definition like:
//  const char opencl_src[] = "preprocessed text of foo.ocl"
// Thus, with gencl, this file says 
//    #include <foo.i>
// and the binary obtained by compiling it
// is fully "baked".  Runtime behavior doesn't depend
// on the contents of some file (e.g., foo.ocl or some
// header that it includes) that might have changed long after this
// file was compiled.
//
// The alternative (USE_GENCL 0) seems to be more along
// the lines of what OpenCL designers imagine.  It makes the text of the
// kernel program the string "#include <foo.c>".  This eliminates
// the need for the extra machinery in gencl.sh, but runtime
// behavior is susceptable to changes in foo.c, or files included
// by foo.c long after this file is compiled.  It also requires some
// hocus pocus to get absolute paths for the -I options needed
// to compile the code at runtime.  Something like:
//  override CFLAGS += -DSRCDIR=\"$(dir $(abspath $<)).\"
#define USE_GENCL 1

#if USE_GENCL
const char *opencl_src = 
#include "kat_opencl_kernel.i"
     ;
#else
#ifndef SRCDIR
#error -DSRCDIR="/absolute/path/to/examples" should have been put on the command-line by GNUmakefile
#endif
#endif

void host_execute_tests(kat_instance *tests, unsigned ntests){
    UCLInfo *infop;
    cl_kernel kern;
    size_t nthreads, tests_sz;
    cl_mem tests_dev;
    const char *kernelname = "dev_execute_tests";
    cl_int err;

#if USE_GENCL
    infop = opencl_init(NULL, opencl_src, "");
#else
    infop = opencl_init(NULL, "#include <kat_dev_execute.h>", 
                        " -I" SRCDIR 
                        " -I" SRCDIR "/../include " 
                        " -DKAT_KERNEL=__kernel "
                        " -DKAT_GLOBAL=__global ");
#endif
    CHECKERR(kern = clCreateKernel(infop->prog, kernelname, &err));
    if (infop->wgsize > 64) infop->wgsize /= 2;
    nthreads = infop->cores * infop->wgsize;
    tests_sz = sizeof(*tests) * (ntests+1); // +1 for sentinel test with method==last
    CHECKERR(tests_dev = clCreateBuffer(infop->ctx, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, tests_sz, tests, &err));
    CHECK(clEnqueueWriteBuffer(infop->cmdq, tests_dev, CL_TRUE, 0, tests_sz, tests, 0, 0, 0));
    CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&tests_dev));
    printf("queuing kernel for %lu threads with %lu work group size, %u tests\n",
	   (unsigned long)nthreads, (unsigned long)infop->wgsize, ntests);
    CHECK(clEnqueueNDRangeKernel(infop->cmdq, kern, 1, 0, &nthreads, &infop->wgsize, 0, 0, 0));
    CHECK(clFinish(infop->cmdq));
    CHECK(clEnqueueReadBuffer(infop->cmdq, tests_dev, CL_TRUE, 0, tests_sz, tests, 0, 0, 0));
    CHECK(clReleaseMemObject(tests_dev));
    CHECK(clReleaseKernel(kern));
    opencl_done(infop);
}
