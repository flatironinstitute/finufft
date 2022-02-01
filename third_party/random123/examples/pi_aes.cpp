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
#include <Random123/aes.h>
#include <stdio.h>
#include <assert.h>
using namespace r123;

// Everyone's favorite PRNG example: calculate pi/4 by throwing darts
// at a square board and counting the fraction that are inside the
// inscribed circle.

// This version uses the C++ API to AESNI.

#include "pi_check.h"
#include "example_seeds.h"

int main(int, char **){
#if R123_USE_AES_NI
    unsigned long seed = example_seed_u32(EXAMPLE_SEED1_U32); // example user-settable seed
    unsigned long hits = 0, tries = 0;
    const int64_t two_to_the_62 = ((int64_t)1)<<62;

    if (!haveAESNI()) {
	std::cerr << "AES-NI instructions not available on this hardware, skipping the pi_aes test." << std::endl;
        return 0;
    }
    typedef AESNI4x32 G;
    G generator;
    // As an example, we illustrate one user-provided seed word and the rest as arbitrary constants
    G::ukey_type ukey = {{(G::ukey_type::value_type)seed, EXAMPLE_SEED2_U32, EXAMPLE_SEED3_U32, EXAMPLE_SEED4_U32}};
    // The key_type constructor transforms the 128bit AES ukey_type to an expanded (1408bit) form.
    G::key_type key = ukey;
    // start ctr from an arbitrary point.  0 would work fine too, this is just to show that it does
    // not matter where it starts from (or for that matter, whether it increments by 1, or some other
    // arbitrary stride or in some other way, as long as it never repeats a key,ctr combination)
    G::ctr_type ctr = {{EXAMPLE_SEED5_U32, EXAMPLE_SEED6_U32, EXAMPLE_SEED7_U32, EXAMPLE_SEED8_U32}};

    printf("Throwing %lu darts at a square board using AESNI4x32\n", NTRIES);
    std::cout << "Initializing AES key with hex userkey: " << std::hex << ukey << " ctr: " << ctr << std::endl;

    while(tries < NTRIES){
        ctr.incr();
        G::ctr_type r = generator(ctr, key);
	if (tries == 0) {
	    std::cout << "first random from AESNI is " << std::hex << r << std::endl;;
	}
        for(size_t j=0; j<r.size(); j+=2){
            int64_t x = (int32_t)r[j];
            int64_t y = (int32_t)r[j+1];
            if( (x*x + y*y) < two_to_the_62 )
                hits++;
            tries++;
        }
    }
    return pi_check(hits, tries);
#else
    std::cout << "AESNI RNG not compiled into this binary, skipping the pi_aes test.\n";
    return 0;	// Not a failure to not have AESNI compiled into this.
#endif
}
