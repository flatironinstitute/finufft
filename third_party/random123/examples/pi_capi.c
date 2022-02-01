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
#include <Random123/threefry.h>
#include <stdio.h>
#include <assert.h>

/* Everyone's favorite PRNG example: calculate pi/4 by throwing darts
// at a square board and counting the fraction that are inside the
// inscribed circle.

// This version uses the C API to threefry2x64. */

#include "pi_check.h"

int main(int argc, char **argv){
    unsigned long hits = 0, tries = 0;
    const int64_t two_to_the_62 = ((int64_t)1)<<62;

    threefry2x64_key_t key = {{0, 0}};
    threefry2x64_ctr_t ctr = {{0, 0}};
    enum { int32s_per_counter = sizeof(ctr)/sizeof(int32_t) };
    (void)argc;(void)argv; /* unused  */

    printf("Throwing %lu darts at a square board using threefry2x64\n", NTRIES);

    /* make the most of each bijection by looping over as many
       int32_t's as we can find in the ctr_type. */
    assert( int32s_per_counter%2 == 0 );
    while(tries < NTRIES){
        /* Use a union to avoid strict aliasing issues. */
        union{
            threefry2x64_ctr_t ct;
            int32_t i32[int32s_per_counter];
        }u;
        size_t j;
        /* Don't worry about the 'carry'.  We're not going to loop
           more than 2^64 times. */
        ctr.v[0]++;
        u.ct = threefry2x64(ctr, key);
        for(j=0; j<int32s_per_counter; j+=2){
            int64_t x = u.i32[j];
            int64_t y = u.i32[j+1];
            if( (x*x + y*y) < two_to_the_62 )
                hits++;
            tries++;
        }
    }
    return pi_check(hits, tries);
}
