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
#ifndef TIME_RANDOM123_H__
#define TIME_RANDOM123_H__ 1

/*
 * This file contains the performance timing kernels for Random123
 * RNGs and a few conventional PRNGs, which have been hacked into
 * this framework.  This code should NOT be considered as an
 * example of using PRNGs.  A few macros try to keep it
 * cross-platform across C (serial or threads), CUDA and OpenCL.
 * The TEST_TPL + include util_expandtpl trick is used to
 * template-generate kernels for all the different RNGs across
 * NxW.
 */

#include <Random123/philox.h>
#include <Random123/threefry.h>

#if defined(__OPENCL_VERSION__)
#define KERNEL __kernel
#define MEMTYPE __global
#define xprintf(x)
#elif defined(__CUDA_ARCH__)
#define xprintf(x)
#else
#define xprintf(x) printf x
#endif

#ifndef KERNEL
#define KERNEL
#endif

#ifndef MEMTYPE
#define MEMTYPE
#endif

/* LOOK_AT forces the compiler to actually produce the code that
   computes the elements of A.  Without it, optimizing compilers can
   elide some or all of the computation of A (i.e., the RNG we're
   trying to get timings for).  There are many ways to do this.  A
   perhaps more natural way would be to keep a running sum of all the
   results so far, but we found that gcc 4.5 and 4.6 could unroll that
   code into a fully SSE-ized loop, which appears to be
   unrepresentative of the kinds of optimizations that are possible in
   practice.  I.e., we could not find any "real" use of the output of
   the RNG that permitted SSE-ization of the RNG. */
#define LOOK_AT(A, I, N) do{                                            \
    if (N==4) if(R123_BUILTIN_EXPECT(!(A.v[N>2?3:0]^A.v[N>2?2:0]^A.v[N>1?1:0]^A.v[0]), 0)) ++I; \
    if (N==2) if(R123_BUILTIN_EXPECT(!(A.v[N>1?1:0]^A.v[0]), 0)) ++I;   \
    if (N==1) if(R123_BUILTIN_EXPECT(!(A.v[0]), 0)) ++I;                \
    }while(0)


/* Macro that will expand later into all the Random123 PRNGs for NxW_R */
/* XXX AMDAPPSDK 2.4 seemed unhappy with the first arg being uint, but
   was ok when it was changed to uint64_t.  It's now back to unsigned
   because that seems more correct and generic.  Nobody's using 2.4
   any more, are they?  Note that this macro is expanded into CPU, 
   CUDA and OpenCL "kernels", so it has to be generic. */
#define TEST_TPL(NAME, N, W, R)                                         \
KERNEL void test_##NAME##N##x##W##_##R(unsigned n, NAME##N##x##W##_ctr_t ctrinit, NAME##N##x##W##_ukey_t uk, MEMTYPE NAME##N##x##W##_ctr_t *ctr) \
{                                                                       \
    unsigned tid = get_global_id(0);                                    \
    unsigned i;                                                         \
    NAME##N##x##W##_ctr_t c, v={{0}};                                   \
    NAME##N##x##W##_key_t k=NAME##N##x##W##keyinit(uk);                 \
    c = ctrinit;                                                        \
    if( R == NAME##N##x##W##_rounds ){                                  \
        for (i = 0; i < n; ++i) {                                       \
	    v = NAME##N##x##W(c, k);                                    \
	    LOOK_AT(v, i, N);                                           \
            c.v[0]++;                                                   \
        }                                                               \
    }else {                                                             \
        for (i = 0; i < n; ++i) {                                       \
            v = NAME##N##x##W##_R(R, c, k);                             \
	    /*xprintf(("1: %s k[0] %lx c[0] %lx v[0] %lx\n", #NAME #N "x" #W "_" #R, (unsigned long) k.v[0], (unsigned long) c.v[0], (unsigned long) v.v[0]));*/ \
	    /*if (c.v[0] == 0) printline_##NAME##N##x##W##_##R(k, c, &v, 1);*/ \
	    LOOK_AT(v, i, N);                                           \
            c.v[0]++;                                                   \
        }                                                               \
    }                                                                   \
    ctr[tid] = v;                                                       \
}

/* Now expand TEST_TPL for all the relevant RNGs */
#include "util_expandtpl.h"

#endif /* TIME_RANDOM123_H__ */
