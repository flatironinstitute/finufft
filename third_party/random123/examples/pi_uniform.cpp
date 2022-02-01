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
#include <stdio.h>
#include <Random123/threefry.h>
#include <Random123/uniform.hpp>

/* Compute pi, using the u01 conversion with threefry2x64 and threefry2x32 */

#include "pi_check.h"
#include "example_seeds.h"

using namespace r123;

template<typename Ftype, typename CBRNG>
void pi(typename CBRNG::key_type k);

int errs = 0;
int main(int, char **){
    uint64_t seed64 = example_seed_u64(EXAMPLE_SEED1_U64); // example user-settable seed
    unsigned long hits = 0, tries = 0;

    // First, we demonstrate how to compute pi
    // using uneg11 to convert the integer output
    // of threefry2x64 to a double in (-1, 1).
    Threefry2x64::ctr_type c = {{0}}, r;
    Threefry2x64::ukey_type uk = {{seed64}};
    Threefry2x64::key_type k = uk;
    printf("%lu uniform doubles from threefry2x64\n", NTRIES);
    while (tries < NTRIES) {
            double x, y;
            c.v[0]++; /* increment the counter */
	    r = threefry2x64(c, k);
            x = uneg11<double>(r.v[0]);
            y = uneg11<double>(r.v[1]);
            if( x*x + y*y < 1.0 )
                hits++;
	    tries++;
    }
    errs += pi_check(hits, tries);

    // Extra credit: use some template hackery to exercise various
    // combinations of float, double and long double, unit64_t and
    // uint32_t and the conversion functions u01, uneg11 and ufixed01.
    // This provides minimal testing of the conversion functions.
    pi<float, Threefry2x64>(k);
    pi<double, Threefry2x64>(k);
    pi<long double, Threefry2x64>(k);
    uint32_t seed32 = example_seed_u32(EXAMPLE_SEED9_U32);

    Threefry2x32::ukey_type ukh = {{seed32}};
    Threefry2x32::key_type kh = ukh;
    pi<float, Threefry2x32>(kh);
    pi<double, Threefry2x32>(kh);
    pi<long double, Threefry2x32>(kh);

    return !!errs;
}

template<typename Ftype, typename CBRNG>
void pi(typename CBRNG::key_type k){
    unsigned long hits = 0, tries = 0;
    CBRNG rng;

    printf("Compute pi with uneg11:\n");
    typename CBRNG::ctr_type c = {{0}}, r;
    hits = tries = 0;
    while (tries < NTRIES) {
        Ftype x, y;
        c.v[0]++; /* increment the counter */
        r = rng(c, k);
        // x and y in the entire square from (-1,-1) to (1,1)
        x = uneg11<Ftype>(r.v[0]);
        y = uneg11<Ftype>(r.v[1]);
        if( x*x + y*y < 1.0 )
            hits++;
        tries++;
    }
    errs += pi_check(hits, tries);

#if __cplusplus >= 201103L
    printf("Compute pi with uneg11all (requires C++11):\n");
    hits = tries = 0;
    while (tries < NTRIES) {
        c.v[0]++; /* increment the counter */
        r = rng(c, k);
        // x and y in the entire square from (-1,-1) to (1,1)
        auto a = uneg11all<Ftype>(r);
        if( a[0]*a[0] + a[1]*a[1] < 1.0 )
            hits++;
        tries++;
    }
    errs += pi_check(hits, tries);
#endif

    printf("Compute pi with u01:\n");
    hits = tries = 0;
    while (tries < NTRIES) {
            Ftype x, y;
            c.v[0]++; /* increment the counter */
	    r = rng(c, k);
            // generate x and y in the first quadrant from (0,0) to (1,1)
            x = u01<Ftype>(r.v[0]);
            y = u01<Ftype>(r.v[1]);
            if( x*x + y*y < 1.0 )
                hits++;
	    tries++;
    }
    errs += pi_check(hits, tries);

    printf("Compute pi with u01fixedpt:\n");
    hits = tries = 0;
    while (tries < NTRIES) {
            Ftype x, y;
            c.v[0]++; /* increment the counter */
	    r = rng(c, k);
            // generate x and y in the first quadrant from (0,0) to (1,1)
            x = u01fixedpt<Ftype>(r.v[0]);
            y = u01fixedpt<Ftype>(r.v[1]);
            if( x*x + y*y < 1.0 )
                hits++;
	    tries++;
    }
    errs += pi_check(hits, tries);
 }

