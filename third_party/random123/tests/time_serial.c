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
 * Single-core CPU test and timing harness for Random123 RNGs.
 * Uses macros and util_expandtpl.h to "templatize" over all the
 * different permutations of RNGs and NxW and R.
 */
#include "util.h"

#include "Random123/philox.h"
#include "Random123/threefry.h"
#include "Random123/ars.h"
#include "Random123/aes.h"

#include "time_misc.h"
#include "util_print.h"

#include "util.h"
#define KERNEL R123_STATIC_INLINE
#define get_global_id(i)    (i)
#include "time_random123.h"

static double cpu_hz = -1.;

#define TEST_TPL(NAME, N, W, R) \
void NAME##N##x##W##_##R(NAME##N##x##W##_ctr_t ctr, NAME##N##x##W##_ukey_t ukey, NAME##N##x##W##_ctr_t kactr, unsigned count, void* unused) \
{ \
    const char *kernelname = #NAME #N "x" #W "_" #R; \
    double cur_time; \
    int n, niterations = numtrials; /* we make niterations + 2 (warmup, overhead) calls to the kernel */ \
    unsigned kcount = 0; \
    double basetime = 0., dt = 0., mindt = 0.; \
    NAME##N##x##W##_ctr_t C, *hC = &C; \
    (void)unused; /* suppress warning */        \
    \
    for (n = -2; n < niterations; n++) { \
	if (n == -2) { \
	    if (count == 0) { \
		/* try to set a good guess for count */ \
		count = 1000000; \
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
		    if(!debug) exit(1);                                 \
                    else break;                                         \
		} else { \
		    dprintf(("%s matched word %d\n", kernelname, xj)); \
		} \
	    } \
	    basetime = dt; \
	    if (debug||verbose) { \
		dprintf(("%s %.3f secs\n", kernelname, basetime)); \
		printline_##NAME##N##x##W##_##R(ukey, ctr, hC, 1); \
	    } \
	    kcount = count + 1; \
	} \
	/* calling timer *before* dprintf avoids an ARMv7 gcc 4.8.3 -O3 compiler bug! */ \
	(void)timer(&cur_time); \
	dprintf(("call function %s\n", kernelname)); \
	test_##NAME##N##x##W##_##R(kcount, ctr, ukey, hC); \
	dt = timer(&cur_time); \
	dprintf(("iteration %d took %.3f secs\n", n, dt)); \
	ALLZEROS(hC, 1, N); \
	if (n == 0 || dt < mindt) mindt = dt; \
    } \
    if (count > 1) { \
	double tpB = (mindt - basetime) / ( (kcount - 1.) * (N * W / 8.) ); \
        if(cpu_hz > 0.) \
            printf("%-17s %#5.3g cpB %#5.3g GB/s %u B granularity (best %u in %.3f s - %.6f s)\n", \
                   kernelname, tpB*cpu_hz, 1e-9/tpB,                    \
                   (unsigned)(N*W/8), kcount, mindt, basetime );        \
        else                                                            \
            printf("%-17s %#5.3g GB/s %u B granularity (best %u in %.3f s - %.6f s)\n", \
                   kernelname, 1e-9/tpB,                                \
                   (unsigned)(N*W/8), kcount, mindt, basetime );        \
	fflush(stdout); \
    } \
}

#include "util_expandtpl.h"

int main(int argc, char **argv)
{
    char *cp;
    unsigned count = 0;
    int keyctroffset = 0;
    void* infop = NULL;
    
    progname = argv[0];
    if (argc > 3|| (argv[1] && argv[1][0] == '-')) {
	fprintf(stderr, "Usage: %s [COUNT]]\n", progname);
	exit(1);
    }
    if (argc > 1)
	count = atoi(argv[1]);
    if ((cp = getenv("TIME_SERIAL_CPU_GHZ")) != NULL) {
        cpu_hz = 1.e9 * atof(cp);
    }
    if ((cp = getenv("TIME_SERIAL_VERBOSE")) != NULL) {
	verbose = atoi(cp);
    }
    if ((cp = getenv("TIME_SERIAL_DEBUG")) != NULL) {
	debug = atoi(cp);
    }
    if ((cp = getenv("TIME_SERIAL_OFFSET")) != NULL) {
	keyctroffset = atoi(cp);
    }
    if ((cp = getenv("TIME_SERIAL_NUMTRIALS")) != NULL) {
        numtrials = atoi(cp);
    }
    if ((cp = getenv("TIME_SERIAL_SEC_PER_TRIAL")) != NULL) {
        sec_per_trial = atof(cp);
    }
#   include "time_initkeyctr.h"
    return 0;
}
