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
/*
 * CUDA test and timing harness for Random123 RNGs.  Uses macros
 * and util_expandtpl.h to "templatize" over all the different
 * permutations of RNGs and NxW and R.
 */

#include "util_cuda.h"

#include "Random123/philox.h"
#include "Random123/threefry.h"

#include "time_misc.h"
#include "util_print.h"

#define KERNEL __global__
#define get_global_id(i) (blockDim.x * blockIdx.x + threadIdx.x)
#include "time_random123.h"

#define TEST_TPL(NAME, N, W, R) \
void NAME##N##x##W##_##R(NAME##N##x##W##_ctr_t ctr, NAME##N##x##W##_ukey_t ukey, NAME##N##x##W##_ctr_t kactr, unsigned count, CUDAInfo *tp) \
{ \
    const char *kernelname = PREFIX #NAME #N "x" #W "_" #R; \
    NAME##N##x##W##_ctr_t *hC, *dC; \
    int n, niterations = numtrials; /* we make niterations + 2 (warmup, overhead) calls to the kernel */ \
    double cur_time; \
    size_t i; \
    const size_t nworkitems = tp->blocks_per_grid * tp->threads_per_block; \
    const size_t szC = nworkitems*sizeof(hC[0]); \
    \
    /* allocate vector of counters in device memory, initialize to zero */ \
    CHECKCALL(cudaMalloc(&dC, szC)); \
    CHECKCALL(cudaMemset((void *)dC, 0, szC)); \
    /* allocate and initialize vector of counters in host memory */ \
    CHECKNOTZERO(hC = (NAME##N##x##W##_ctr_t *)malloc(szC)); \
    for (i = 0; i < nworkitems; i++) { \
	size_t xi; \
	for (xi = 0; xi < N; xi++) { \
	    hC[i].v[xi] = 0; \
	} \
    } \
    unsigned kcount = 0; \
    double basetime = 0., dt = 0., mindt = 0.; \
    for (n = -2; n < niterations; n++) { \
	if (n == -2) { \
	    if (count == 0) { \
		/* try to set a good guess for count */ \
		count = (unsigned)(tp->cycles ? tp->cycles * 1e-8 : 10000); \
		dprintf(("starting with count = %u\n", count)); \
	    } \
	    kcount = count; \
	} else if (n == -1) { \
	    /* use first iteration time to calibrate count to get approximately sec_per_trial */ \
	    if (count > 1) { \
		count = (unsigned)(count * sec_per_trial / dt); \
		dprintf(("scaled count = %u\n", count)); \
	    } \
	    /* second iteration is to calculate overhead after warmup */ \
	    kcount = 1; \
	} else if (n == 0) { \
	    int xj; \
	    /* Check that we got the expected value */ \
	    for (xj = 0; xj < N; xj++) { \
		if (kactr.v[xj] != hC[0].v[xj]) { \
		    printf("%s mismatch: xj = %d, expected\n", kernelname, xj); \
		    printline_##NAME##N##x##W##_##R(ukey, ctr, &kactr, 1); \
		    printf("    but got\n"); \
		    printline_##NAME##N##x##W##_##R(ukey, ctr, hC, 1); \
                    if(!debug) exit(1); \
                    else break; \
		} else { \
		    dprintf(("%s matched word %d\n", kernelname, xj)); \
		} \
	    } \
	    basetime = dt; \
	    if (debug||verbose) { \
		dprintf(("%s %.3f secs for %lu workitems test on device %d (%s)\n", \
		       kernelname, basetime, (unsigned long) nworkitems, \
		       tp->devnum, tp->dev.name)); \
		printline_##NAME##N##x##W##_##R(ukey, ctr, hC, (verbose < 2) ? 1 : nworkitems); \
	    } \
	    kcount = count + 1; \
	} \
	dprintf(("launch %s\n", kernelname)); \
	(void)timer(&cur_time); \
	test_##NAME##N##x##W##_##R<<<tp->blocks_per_grid, tp->threads_per_block>>>(kcount, ctr, ukey, dC); \
	dprintf(("synchronize\n")); \
	CHECKCALL(cudaDeviceSynchronize()); \
	dprintf(("copy results back from device to host\n")); \
	CHECKCALL(cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost)); \
	dt = timer(&cur_time); \
	dprintf(("iteration %d took %.3f secs\n", n, dt)); \
	ALLZEROS(hC, nworkitems, N); \
	if (n == 0 || dt < mindt) mindt = dt; \
    } \
    if (count > 1) { \
	double tpB = (mindt - basetime) / ( (kcount - 1.) * nworkitems * (N * W / 8.) ); \
	printf("%-17s %#5.3g cpB, %#5.3g GB/s on dev %d %u B granularity (best %u in %.3f s - %.6f s)\n", \
	       kernelname + sizeof(PREFIX) - 1, \
	       tpB * tp->cycles , 1e-9/tpB, \
	       tp->devnum, (unsigned)(N*W/8), kcount, mindt, basetime ); \
	fflush(stdout); \
    } \
    /* free host and device memory */ \
    free(hC); \
    CHECKCALL(cudaFree(dC)); \
}

#include "util_expandtpl.h"

int main(int argc, char **argv)
{
    char *cp;
    unsigned count = 0;
    CUDAInfo *infop;
    int keyctroffset = 0;
    
    progname = argv[0];
    if (argc > 3|| (argv[1] && argv[1][0] == '-')) {
	fprintf(stderr, "Usage: %s [COUNT [DEVICESTRING]]\n", progname);
	exit(1);
    }
    if (argc  > 1)
	count = atoi(argv[1]);
    if ((cp = getenv("TIME_CUDA_VERBOSE")) != NULL) {
	verbose = atoi(cp);
    }
    if ((cp = getenv("TIME_CUDA_DEBUG")) != NULL) {
	debug = atoi(cp);
    }
    if ((cp = getenv("TIME_CUDA_OFFSET")) != NULL) {
	keyctroffset = atoi(cp);
    }
    if ((cp = getenv("TIME_CUDA_NUMTRIALS")) != NULL) {
        numtrials = atoi(cp);
    }
    if ((cp = getenv("TIME_CUDA_SEC_PER_TRIAL")) != NULL) {
        sec_per_trial = atof(cp);
    }
    infop = cuda_init(argc > 2 ? argv[2] : NULL);
    /* define macro to initialize and call the kernels */
#   include "time_initkeyctr.h"
    cuda_done(infop);
    return 0;
}
