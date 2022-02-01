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
#include <gsl/gsl_randist.h>
#include <stdio.h>
#include "Random123/philox.h"
#include "Random123/threefry.h"
#include "Random123/gsl_microrng.h"

/* Compute pi, using the gsl_ran_flat distribution with
   an underlying threefry4x64 counter-based rng (cbrng).
   We can call cbrng 8 times between calls to cbrng_reset */

GSL_MICRORNG(cbrng, threefry4x64); /* creates gsl_rng_cbrng */

#include "pi_check.h"

int main(int argc, char **argv){
    unsigned long hits = 0, tries = 0;
    gsl_rng *r;
    (void)argc; (void)argv; /* unused */

    threefry4x64_ctr_t c = {{0}};
    threefry4x64_key_t k = {{0}};
    r = gsl_rng_alloc(gsl_rng_cbrng);
    printf("%lu uniforms from %s\n", NTRIES, gsl_rng_name(r));
    while (tries < NTRIES) {
            double x, y;
            c.v[0]++; /* increment the counter */
            cbrng_reset(r, c, k); /* reset the rng to the new counter */
            x = gsl_ran_flat (r, -1.0, 1.0);
            y = gsl_ran_flat (r, -1.0, 1.0);
            if( x*x + y*y < 1.0 )
                hits++;
	    tries++;
    }
    gsl_rng_free (r);
    return pi_check(hits, tries);
}
