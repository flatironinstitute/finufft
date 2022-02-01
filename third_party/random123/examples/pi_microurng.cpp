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
// Everyone's favorite PRNG example: calculate pi/4 by throwing darts
// at a square board and counting the fraction that are inside the
// inscribed circle.

// This version uses Philox4x32 with a MicroURNG and the C++11 standard
// library std::uniform_real distribution to generate floats in [-1..1]

// N.B.  The results are hardware dependent even though the underlying
// counter based RNG is hardware and endian-invariant.  On x86,
// floating point temporaries, e.g., x, y, x*x, etc., are stored in
// 80-bit extended precision registers.  On x86-64 (and other IEEE-754
// systems), temporaries are stored in 32-bit SSE registers.

#include <Random123/philox.h>
#include <Random123/MicroURNG.hpp>
#include <Random123/ReinterpretCtr.hpp>
#if R123_USE_CXX11_RANDOM
#include <random>
#endif
#include <iostream>
#include <iomanip>
#include "pi_check.h"

using namespace r123;

int main(int, char**){
    typedef Philox4x32 RNG;
    RNG::ctr_type c = {{}};
    RNG::key_type k = {{}};
    MicroURNG<RNG> longmurng(c.incr(), k);
#if R123_USE_STD_RANDOM
    std::uniform_real_distribution<float> u(-1., 1.);

    // First, compute pi with a nice long MicroURNG that we cancall
    // billions of times (2^31) before it runs out of state:
    unsigned long hits=0;
    std::cout << "Calling a single MicroURNG " << NTRIES << " times" << std::endl;
    for(unsigned long i=0; i<NTRIES; ++i){
        float x = u(longmurng);
        float y = u(longmurng);
        if( (x*x + y*y) < 1.0f )
            hits++;
    }
    if (pi_check(hits, NTRIES) != 0) {
	return 1;
    }
    // MicroURNGs are very light-weight.  It shouldn't be
    // too expensive to create a new one every time through the loop:
    std::cout << "Creating and calling a new MicroURNG " << NTRIES << " times" << std::endl;
    hits=0;
    for(unsigned long i=0; i<NTRIES; ++i){
        MicroURNG<RNG> shorturng(c.incr(), k);
        float x = u(shorturng);
        float y = u(shorturng);
        if( (x*x + y*y) < 1.0f )
            hits++;
    }
    return pi_check(hits, NTRIES);
#else
    // MicroURNG's are interesting because they allow us to use std::distributions,
    // as in the above code.  Std::distributions are nice, but if all we need is
    // a uniform integer, we can do without such fancy C++11 features:
    unsigned long hits=0;
    std::cout << "Calling a single MicroURNG " << NTRIES << " times" << std::endl;
    for(unsigned long i=0; i<NTRIES; ++i){
        float x = 2.*longmurng()/(double)std::numeric_limits<uint32_t>::max() - 1.;
        float y = 2.*longmurng()/(double)std::numeric_limits<uint32_t>::max() - 1.;
        if( (x*x + y*y) < 1.0f )
            hits++;
    }
    if (pi_check(hits, NTRIES) != 0) {
	return 1;
    }
#endif
    

}
