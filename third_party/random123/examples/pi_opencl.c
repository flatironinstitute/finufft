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
// Simple OpenCL device kernel and host main program to
// compute pi via random darts at a square

// util_opencl.h has a large amount of OpenCL boilerplate.
// It contains nothing RNG-specific.
#include "../tests/util_opencl.h"
#include "pi_check.h"
#include "example_seeds.h"

static const char *opencl_src = 
// pi_opencl_kernel.i contains a literal string
// with the kernel source code.  It's generated
// by ./gencl.sh opencl_kernel.ocl
#include "pi_opencl_kernel.i"   
     ;

const char *progname;
int verbose = 0;
int debug = 0;

int
main(int argc, char **argv)
{
    unsigned seed = example_seed_u32(EXAMPLE_SEED9_U32); // example user-settable seed
    unsigned count = argc > 1 ? atoi(argv[1]) : 0;
    UCLInfo *infop;
    size_t i, nthreads, hits_sz;
    cl_mem hits_dev;
    cl_uint2 *hits_host;
    const char *kernelname = "counthits";
    cl_int err;
    cl_kernel kern;
    double d = 0.;

    d = timer(&d);
    progname = argv[0];
    verbose = debug = argc > 2 ? atoi(argv[2]): 0;
    infop = opencl_init(argc > 3 ? argv[3] : NULL, opencl_src, argc > 4 ? argv[4] : "");
    CHECKERR(kern = clCreateKernel(infop->prog, kernelname, &err));
    if (infop->wgsize > 64) infop->wgsize /= 2;
    nthreads = infop->cores * infop->wgsize;
    if (count == 0)
	count = NTRIES/nthreads;
    hits_sz = nthreads * sizeof(hits_host[0]);
    CHECKNOTZERO(hits_host = (cl_uint2 *)malloc(hits_sz));
    CHECKERR(hits_dev = clCreateBuffer(infop->ctx, CL_MEM_WRITE_ONLY, hits_sz, 0, &err));
    CHECK(clSetKernelArg(kern, 0, sizeof(unsigned), (void*)&count));
    CHECK(clSetKernelArg(kern, 1, sizeof(unsigned), (void*)&seed));
    CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), (void*)&hits_dev));
    printf("queuing kernel for %lu threads with %lu work group size, %u points with seed 0x%x\n",
	   (unsigned long)nthreads, (unsigned long)infop->wgsize, count, seed);
    CHECK(clEnqueueNDRangeKernel(infop->cmdq, kern, 1, 0, &nthreads, &infop->wgsize, 0, 0, 0));
    CHECK(clFinish(infop->cmdq));
    CHECK(clEnqueueReadBuffer(infop->cmdq, hits_dev, CL_TRUE, 0, hits_sz, hits_host, 0, 0, 0));

    unsigned long hits = 0, tries = 0;
    for (i = 0; i < nthreads; i++) {
	if (debug)
	    printf("%lu %u %u\n", (unsigned long)i, hits_host[i].x, hits_host[i].y);
	hits += hits_host[i].x;
	tries += hits_host[i].y;
    }
    CHECK(clReleaseMemObject(hits_dev));
    CHECK(clReleaseKernel(kern));
    opencl_done(infop);
    return pi_check(hits, tries);
}
