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
#include <Random123/ReinterpretCtr.hpp>
#include <stdio.h>
#include <assert.h>

// Everyone's favorite PRNG example: calculate pi/4 by throwing darts
// at a square board and counting the fraction that are inside the
// inscribed circle.

// This version uses the C++ API to Threefry4x64, and the
// ReinterpretCtr template to get 32-bit values.  

// Note - by using ReinterpretCtr, the result depends on the
// endianness of the hardware it runs on even though the underlying
// generator is endian-independent.  An easy way to make the result
// endian-independent would be to eliminate ReinterpretCtr and to use
// a generator that works natively with 32-bit quantities, e.g.,
// Threefry4x32 or Philox4x32.

using namespace r123;

#include "pi_check.h"

int main(int, char **){
    unsigned long hits = 0, tries = 0;
    const int64_t two_to_the_62 = ((int64_t)1)<<62;

    typedef ReinterpretCtr<r123array8x32, Threefry4x64> G;
    G generator;
    G::key_type key = {{}}; // initialize with zeros
    G::ctr_type ctr = {{}};

    printf("Throwing %lu darts at a square board using Threefry4x64\n", NTRIES);

    while(tries < NTRIES){
        ctr.incr();
        G::ctr_type r = generator(ctr, key);
        for(size_t j=0; j<r.size(); j+=2){
            int64_t x = (int32_t)r[j];
            int64_t y = (int32_t)r[j+1];
            if( (x*x + y*y) < two_to_the_62 )
                hits++;
            tries++;
        }
    }
    return pi_check(hits, tries);
}
