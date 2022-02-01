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
#ifndef UTIL_PRINT_H__
#define UTIL_PRINT_H__

#include <stdio.h>

extern int verbose;

#define TEST_TPL(NAME, N, W, R) \
void printline_##NAME##N##x##W##_##R(NAME##N##x##W##_ukey_t ukey, NAME##N##x##W##_ctr_t ictr, NAME##N##x##W##_ctr_t *octrs, size_t nctr) \
{ \
    size_t i; \
    for (i = 0; i < nctr; i++) { \
	if (i > 0) { \
	    printf("  [%lu]", (unsigned long)i); \
	    PRINTARRAY(octrs[i], stdout); \
	    putc('\n', stdout); \
	    fflush(stdout); \
	} else { \
	    PRINTLINE(NAME, N, W, R, ictr, ukey, octrs[0], stdout); \
	} \
	if (verbose < 2) break; \
    } \
}

#include "util_expandtpl.h"

#endif /* UTIL_PRINT_H__ */
