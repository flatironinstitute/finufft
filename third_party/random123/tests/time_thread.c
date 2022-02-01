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
 * Pthreads test and timing harness for Random123 RNGs.
 * Uses macros and util_expandtpl.h to "templatize" over all the
 * different permutations of RNGs and NxW and R.
 */

#include "util.h"
#include <sys/stat.h>

#include "Random123/philox.h"
#include "Random123/threefry.h"
#include "Random123/ars.h"
#include "Random123/aes.h"

#include "time_misc.h"
#include "util_print.h"
#include "util.h"
#include <pthread.h>

/*
 * Main thread initializes thread_info[i].started to zero, .tid to its own
 * pthread_self() as a placeholder.
 * Child #i sets thread_info[i].tid to pthread_self() and then
 * sets .started.  Only child #i ever writes to thread_info[i], and
 * does so only once.  So searching through thread_info is race-free
 * for get_global_id, since it only ever looking for its own pthread_self
 * anyway.  The write to started needs to be atomic.
 * This is all so that we can use the same kernel as OpenCL/CUDA.
 * Note that parent keeps its own copy of thread ids returned
 * by pthread_create in tids[] to avoid any races with thread_info.
 */
typedef struct {
    int started; /* started == 1 means tid contains pthread_self() of thread */
    pthread_t tid;
} ThreadInfo;
static volatile ThreadInfo *thread_info; /* thread_id state, one per thread */ \
static int thread_count = 12;

/* Linear search should be fast enough for small thread count... */
R123_STATIC_INLINE int get_global_id(int x)
{
    int i;
    pthread_t me = pthread_self();
    (void)x; /* why is this an arg? */
    for (i = 0; i < thread_count; i++) {
	if (thread_info[i].started && pthread_equal(me, thread_info[i].tid))
	    return i;
    }
    fprintf(stderr, "could not find thread %lu\n", (unsigned long) me);
    pthread_exit(NULL);
}

#define KERNEL R123_STATIC_INLINE
#include "time_random123.h"

#define TEST_TPL(NAME, N, W, R) \
typedef struct { \
    unsigned kcount; \
    NAME##N##x##W##_ukey_t ukey; \
    NAME##N##x##W##_ctr_t ctr; \
    NAME##N##x##W##_ctr_t *octrs; \
} ThreadData_##NAME##N##x##W##_##R; \
\
typedef struct { \
    ThreadData_##NAME##N##x##W##_##R *tp; \
    volatile ThreadInfo *tip; \
} ThreadArg_##NAME##N##x##W##_##R; \
\
/* thread_run is launched in a new thread by pthread_create */ \
void *thread_run_##NAME##N##x##W##_##R(void *p) \
{ \
    ThreadArg_##NAME##N##x##W##_##R *ta = p; \
    ThreadData_##NAME##N##x##W##_##R *tp = ta->tp; \
    volatile ThreadInfo *tip = ta->tip; \
    /* store our thread id for use by get_global_info */ \
    tip->tid = pthread_self(); \
    tip->started = 1; \
    test_##NAME##N##x##W##_##R(tp->kcount, tp->ctr, tp->ukey, tp->octrs); \
    return tp; \
}\
void NAME##N##x##W##_##R(NAME##N##x##W##_ctr_t ctr, NAME##N##x##W##_ukey_t ukey, NAME##N##x##W##_ctr_t kactr, unsigned count, void* unused) \
{ \
    const char *kernelname = #NAME #N "x" #W "_" #R; \
    double cur_time; \
    int i, n, niterations = numtrials; /* we make niterations + 2 (warmup, overhead) calls to the kernel */ \
    double basetime = 0., dt = 0., mindt = 0.; \
    ThreadData_##NAME##N##x##W##_##R td; /* same for all threads */ \
    ThreadArg_##NAME##N##x##W##_##R *tap; /* args for thread_run, 1 per thread */ \
    pthread_t me = pthread_self(); /* parent thread id */ \
    pthread_t *tids; /* array of child thread ids */ \
    void *vp; /* return from join */ \
    (void)unused; /* suppress warning */                                \
    CHECKNOTZERO(thread_info = (ThreadInfo *) malloc(sizeof(thread_info[0])*thread_count)); \
    CHECKNOTZERO(tap = (ThreadArg_##NAME##N##x##W##_##R *) malloc(sizeof(tap[0])*thread_count)); \
    CHECKNOTZERO(tids = (pthread_t *) malloc(sizeof(tids[0])*thread_count)); \
    for (i = 0; i < thread_count; i++) { \
	thread_info[i].started = 0; \
	thread_info[i].tid = me; \
	tap[i].tip = &thread_info[i]; \
	tap[i].tp = &td; \
    } \
    CHECKNOTZERO(td.octrs = (NAME##N##x##W##_ctr_t *) malloc(sizeof(td.octrs[0])*thread_count)); \
    td.ukey = ukey; \
    td.ctr = ctr; \
    td.kcount = 0; \
    for (n = -2; n < niterations; n++) { \
	if (n == -2) { \
	    if (count == 0) { \
		/* try to set a good guess for count */ \
		count = 1000000; \
		dprintf(("starting with count = %u\n", count)); \
	    } \
	    td.kcount = count; \
	} else if (n == -1) { \
	    /* use first iteration time to calibrate count to get approximately sec_per_trial */ \
	    if (count > 1) { \
		count = (unsigned)(count * sec_per_trial / dt); \
		dprintf(("scaled count = %u\n", count)); \
	    } \
	    /* second iteration is to calculate overhead after warmup */ \
	    td.kcount = 1; \
	} else if (n == 0) { \
	    int xj; \
	    /* Check that we got the expected value */ \
	    for (xj = 0; xj < N; xj++) { \
		if (kactr.v[xj] != td.octrs[0].v[xj]) { \
		    printf("%s mismatch: xj = %d, expected\n", kernelname, xj); \
		    printline_##NAME##N##x##W##_##R(ukey, ctr, &kactr, 1); \
		    printf("    but got\n"); \
		    printline_##NAME##N##x##W##_##R(ukey, ctr, td.octrs, 1); \
                    if(!debug) exit(1);                                 \
                    else break;                                         \
		} else { \
		    dprintf(("%s matched word %d\n", kernelname, xj)); \
		} \
	    } \
	    basetime = dt; \
	    if (verbose) { \
		dprintf(("%s %.3f secs\n", kernelname, basetime)); \
		printline_##NAME##N##x##W##_##R(ukey, ctr, td.octrs, thread_count); \
	    } \
	    td.kcount = count + 1; \
	} \
	dprintf(("call function %s\n", kernelname)); \
	(void)timer(&cur_time); \
	for (i = 0; i < thread_count; i++) { \
	    CHECKZERO(pthread_create(&tids[i], NULL, thread_run_##NAME##N##x##W##_##R, &tap[i])); \
	    dprintf(("thread %d started\n", i)); \
	} \
	for (i = 0; i < thread_count; i++) { \
	    CHECKZERO(pthread_join(tids[i], &vp)); \
	    dprintf(("thread %d done\n", i)); \
	} \
	dt = timer(&cur_time); \
	dprintf(("iteration %d took %.3f secs\n", n, dt)); \
	ALLZEROS(td.octrs, thread_count, N); \
	if (n == 0 || dt < mindt) mindt = dt; \
    } \
    if (count > 1) { \
	double tpB = (mindt - basetime) / ( (td.kcount - 1.) * thread_count * (N * W / 8.) ); \
	printf("%-17s %#5.3g GB/s %u B granularity (best %u in %.3f s - %.6f s)\n", \
	       kernelname, 1e-9/tpB, \
	       (unsigned)(N*W/8), td.kcount, mindt, basetime ); \
	fflush(stdout); \
    } \
    free((void *) thread_info); \
    thread_info = NULL; \
    free(td.octrs); \
    free(tap); \
    free(tids); \
}

#include "util_expandtpl.h"

int main(int argc, char **argv)
{
    char *cp;
    unsigned count = 0;
    int keyctroffset = 0;
    void *infop = NULL;
    
    progname = argv[0];
    if (argc > 3|| (argv[1] && argv[1][0] == '-')) {
	fprintf(stderr, "Usage: %s [COUNT]\n", progname);
	exit(1);
    }
    if (argc > 1)
	count = atoi(argv[1]);
    if ((cp = getenv("TIME_THREAD_NTHREADS")) != NULL) {
        thread_count = atoi(cp);
    }
    printf("Running with %d threads.  Try 'env TIME_THREAD_NTHREADS=N %s' to change it\n", thread_count, argv[0]);

    if ((cp = getenv("TIME_THREAD_VERBOSE")) != NULL) {
	verbose = atoi(cp);
    }
    if ((cp = getenv("TIME_THREAD_DEBUG")) != NULL) {
	debug = atoi(cp);
    }
    if ((cp = getenv("TIME_THREAD_OFFSET")) != NULL) {
	keyctroffset = atoi(cp);
    }
    if ((cp = getenv("TIME_THREAD_NUMTRIALS")) != NULL) {
        numtrials = atoi(cp);
    }
    if ((cp = getenv("TIME_THREAD_SEC_PER_TRIAL")) != NULL) {
        sec_per_trial = atof(cp);
    }
#   include "time_initkeyctr.h"
    return 0;
}
