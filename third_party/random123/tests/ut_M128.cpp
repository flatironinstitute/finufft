/*
Copyright 2010-2016, D. E. Shaw Research.
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
#include <Random123/features/compilerfeatures.h>
#if !R123_USE_SSE
#include <stdio.h>
int main(){ printf("No SSE.  Nothing to check\n"); return 0; }
#else

#include <Random123/features/sse.h>
#include <sstream>

int main(int, char **){
    r123m128i uninitialized;
    __m128i zm = _mm_setzero_si128();
#if R123_USE_CXX1X_UNRESTRICTED_UNIONS
    r123m128i zM(zm);
#else
    r123m128i zM; zM.m = zm;
#endif
    uninitialized.m = _mm_setzero_si128();

    // operator bool (or maybe void*)
    assert(!uninitialized);
    assert(!zM);

    // operator=(__m128i)
    // conversion to __m128i
    __m128i one = _mm_set_epi32(0, 0, 0, 1);
    __m128i two = _mm_set_epi32(0, 0, 0, 2);
    r123m128i One, Two;
    One = one;
    Two = two;
    assert(!!One);
    assert(!!Two);
    r123m128i AnotherOne;
    AnotherOne = one;

    assert( AnotherOne == One );
    assert( Two != One );
    __m128i m = One;
    AnotherOne = m;
    assert( AnotherOne ==  One );

    // operator++ (prefix)
    ++One;
    assert( One == Two );
    assert( One != AnotherOne );

    // operator+=(R123_ULONG_LONG)
    // operator==(R123_ULONG_LONG, r123m128i)
    R123_ULONG_LONG ull = 2;
    AnotherOne += 1;
    for(int i=0; i<1000; ++i){
        AnotherOne += i;
        ull += i;
        for(int j=0; j<i; ++j){
            assert(One != AnotherOne);
            ++One;
        }
        assert(One == AnotherOne);
        assert(ull == AnotherOne);
    }

    // Do some additions that require carrying.
    // Check the identity behavior of the streams
    // as well
#if R123_USE_64BIT
    for(uint64_t i=0; i<1000; ++i){
        uint64_t fff = (~((uint64_t)0)) - i;
        AnotherOne += fff;
        ull += fff; // will overflow
        One += fff/2;
        One += fff - fff/2;
        assert(AnotherOne == One);
        assert( !(ull == One) );
        std::stringstream ss;
        r123m128i YetAnother;
        ss << AnotherOne;
        ss >> YetAnother;

        assert( YetAnother == AnotherOne );
    }
#endif

    // Sep 2011 - clang in the fink build of llvm-2.9.1 on MacOS 10.5.8
    // fails to catch anything, and hence fails this test.  I suspect
    // a problem with the packaging/installation rather than a bug
    // in llvm.  However, if it shows up in other contexts, some
    // kind of #ifndef might be appropriate.  N.B.  There's a similar
    // exception test in ut_carray.cpp
    bool caught;
    caught = false;
    try{
        (void)(One < AnotherOne);
    }catch(std::runtime_error& ){ caught = true; }
    assert(caught);

    caught = false;
    try{
        (void)(One <= AnotherOne);
    }catch(std::runtime_error& ){ caught = true; }
    assert(caught);

    caught = false;
    try{
        (void)(One > AnotherOne);
    }catch(std::runtime_error& ){ caught = true; }
    assert(caught);

    caught = false;
    try{
        (void)(One >= AnotherOne);
    }catch(std::runtime_error& ){ caught = true; }
    assert(caught);

    // assemble_from_u32<r123m128i>

    std::cout << "ut_M128: OK\n";
    return 0;
}

#endif
