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
// Simple CUDA device kernel and host main program to
// compute pi via random darts at a square

// functions for boilerplate CUDA init and done
#include "../tests/util_cuda.h"

#include <Random123/philox.h>

using namespace r123;

int debug = 0;
const char *progname;


// CUDA Kernel:
// generates n x,y points and returns hits[tid] with the count of number
// of those points within the unit circle on each thread.
__global__ void counthits(unsigned n, unsigned useed, uint2 *hitsp)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned hits = 0, tries = 0;
    typedef Philox4x32 G;
    G rng;
    G::key_type k = {{tid, useed}};
    G::ctr_type c = {{}};

    while (tries < n) {
	union {
	    G::ctr_type c;
	    int4 i;
	}u;
	c.incr();
	u.c = rng(c, k);
	int64_t x1 = u.i.x, y1 = u.i.y;
	int64_t x2 = u.i.z, y2 = u.i.w;
	if ((x1*x1 + y1*y1) < (1LL<<62)) {
	    hits++;
	}
	tries++;
	if ((x2*x2 + y2*y2) < (1LL<<62)) {
	    hits++;
	}
	tries++;
    }
    hitsp[tid] = make_uint2(hits, tries);
}

#include "pi_check.h"
#include "example_seeds.h"

int
main(int argc, char **argv)
{
    unsigned seed = example_seed_u32(EXAMPLE_SEED9_U32); // example user-settable seed
    CUDAInfo *infop;
    uint2 *hits_host, *hits_dev;
    size_t hits_sz;
    unsigned nthreads;
    unsigned count = argc > 1 ? atoi(argv[1]) : 0;
    double d  = 0.;

    d = timer(&d);
    progname = argv[0];
    debug = argc > 2 ? atoi(argv[2]): 0;

    infop = cuda_init(argc > 3 ? argv[3] : NULL);
    nthreads =  infop->blocks_per_grid * infop->threads_per_block;
    if (count == 0)
	count = NTRIES/nthreads;

    hits_sz = nthreads * sizeof(hits_host[0]);
    CHECKCALL(cudaMalloc(&hits_dev, hits_sz));
    CHECKNOTZERO((hits_host = (uint2 *)malloc(hits_sz)));

    printf("starting %u blocks with %u threads/block for %u points each with seed 0x%x\n",
	   infop->blocks_per_grid, infop->threads_per_block, count, seed);
    fflush(stdout);

    counthits<<<infop->blocks_per_grid, infop->threads_per_block>>>(count, seed, hits_dev);

    CHECKCALL(cudaDeviceSynchronize());
    CHECKCALL(cudaMemcpy(hits_host, hits_dev, nthreads*sizeof(hits_dev[0]),
		   cudaMemcpyDeviceToHost));

    unsigned long hits = 0, tries = 0;
    for (unsigned i = 0; i < nthreads; i++) {
	if (debug)
	    printf("%u %u %u\n", i, hits_host[i].x, hits_host[i].y);
	hits += hits_host[i].x;
	tries += hits_host[i].y;
    }
    CHECKCALL(cudaFree(hits_dev));
    free(hits_host);
    cuda_done(infop);
    return pi_check(hits, tries);
}
