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
#include <Random123/array.h>
#include <iostream>
#include <typeinfo>
#include <sstream>
#include <limits>
#include <assert.h>
#include <vector>
#include "util_demangle.hpp"

using namespace std;

template<typename T>
inline static T zero(){ return 0; }

template<typename T>
inline static T fff(){ return ~T(0); }

template<typename T>
inline R123_ULONG_LONG ull(const T& t){ return static_cast<R123_ULONG_LONG>(t); }

template <typename T>
inline uint32_t get32(const T& t, size_t n){
    return t>>(n*32);
}

#if R123_USE_SSE
template<>
inline r123m128i zero<r123m128i>(){ r123m128i M; M.m=_mm_setzero_si128(); return M;}

template<>
inline r123m128i fff<r123m128i>(){ r123m128i M; M.m=_mm_set_epi32(~0, ~0, ~0, ~0); return M;}

template<>
inline R123_ULONG_LONG ull<r123m128i>(const r123m128i& t){ 
    return _mm_extract_lo64(t.m);
}

template <>
inline uint32_t get32<r123m128i>(const r123m128i& t, size_t n){
    switch(n){
    case 3: return _mm_cvtsi128_si32(_mm_srli_si128(t.m, 12));
    case 2: return _mm_cvtsi128_si32(_mm_srli_si128(t.m, 8));
    case 1: return _mm_cvtsi128_si32(_mm_srli_si128(t.m, 4));
    }
    return _mm_cvtsi128_si32(t.m);
}
#endif

struct dummySeedSeq{
    typedef uint32_t result_type;
    template <typename ITER>
    void generate(ITER b, ITER e){
        uint32_t v = 0xdeadbeef;
        for(; b!=e; ++b){
            *b = v;
            v += 0xbaddecaf;
        }
    }
};

template <typename AType>
void doit(size_t N, size_t W){
    AType uninitialized;
    typedef AType atype;
    typedef typename atype::value_type vtype;
    typedef typename atype::iterator  itype;

    assert( R123_W(AType) == W );

    cout << "doit<" << demangle(uninitialized) << ">";
    // size
    assert(uninitialized.size() == N);

    // width
    assert(sizeof(vtype)*8 == W);
    // data
    assert(uninitialized.data() == &uninitialized.v[0]);

    // front
    assert(&uninitialized.front() == uninitialized.data());

    // back
    assert(&uninitialized.back() == uninitialized.data()+(N-1));

    // The ut_carray Random123 unit test uses an empty initializer list to
    // construct instances of different r123 arrays, in a test that's
    // templated on array type.  This works fine for all of the r123 array
    // types except r123array1xm128i---i.e., an "array" consisting of a single
    // __m128i value.  GCC defines __m128i as a single long long,
    //
    // typedef long long __m128i __attribute__ ((__vector_size__ (16),
    //                                           __may_alias__));
    //
    // while Intel defines it as a union,
    //
    // typedef union  _MMINTRIN_TYPE(16) __m128i {
    // #if !defined(_MSC_VER)
    //      /*
    //       * To support GNU compatible intialization with initializers list,
    //       * make first union member to be of int64 type.
    //       */
    //      __int64             m128i_gcc_compatibility[2];
    // #endif
    //     /*
    //      * Although we do not recommend using these directly, they are here
    //      * for better MS compatibility.
    //      */
    //     __int8              m128i_i8[16];
    //     __int16             m128i_i16[8];
    //     __int32             m128i_i32[4];
    //     __int64             m128i_i64[2];
    //     unsigned __int8     m128i_u8[16];
    //     unsigned __int16    m128i_u16[8];
    //     unsigned __int32    m128i_u32[4];
    //     unsigned __int64    m128i_u64[2];
    //
    //     /*
    //      * This is what we used to have here alone.
    //      * Leave for backward compatibility.
    //      */
    //     char c[16];
    // } __m128i;
    //
    // but PGI defines __m128i as a struct,
    //
    // typedef struct {
    //   private: long long m128i_i64[2];
    // } __attribute__((aligned(16))) __m128i;
    //
    // which can't be initialized with initializer lists before C++11.

    // constructor with initializer.  [], at
#ifndef __PGI
    AType z = {{}};
#else
    AType z;
    z.fill(zero<vtype>());
#endif
    for(unsigned i=0; i<N; ++i){
        assert(!z[i]);
        assert(!z.at(i));
        uninitialized[i] = z[i];
        uninitialized[i] += (i+1);
    }

    // Copy-assignment
    atype iota = uninitialized;

    // begin/end
    for(itype p=iota.begin(); p!=iota.end(); ++p){
        assert((int)ull(*p) == 1+ (p-iota.begin()));
    }
    // cbegin/cend
    for(typename atype::const_iterator p=iota.cbegin(); p!=iota.cend(); ++p){
        assert((int)ull(*p) == 1+ (p-iota.cbegin()));
    }

    // rbegin/rend
    for(typename atype::reverse_iterator p=iota.rbegin(); p!=iota.rend(); ++p){
        assert((int)ull(*p) == iota.rend()-p);
    }

    // crbegin/crend
    for(typename atype::const_reverse_iterator p=iota.crbegin(); p!=iota.crend(); ++p){
        assert((int)ull(*p) == iota.crend()-p);
    }

    // == and !=
    assert(iota == uninitialized);
    assert(!(iota != uninitialized));
    
    for(size_t i=0; i<N; ++i){
        atype notequal = iota;
        ++notequal[i];
        assert(notequal != iota);
        assert(!(notequal == iota));
    }

    // Sep 2011 - clang in the fink build of llvm-2.9.1 on MacOS 10.5.8
    // fails to catch anything, and hence fails this test.  I suspect
    // a problem with the packaging/installation rather than a bug
    // in llvm.  However, if it shows up in other contexts, some
    // kind of #ifndef might be appropriate.  N.B.  There's another
    // exception test below and one in ut_M128.cpp
    // check that at throws 
    bool caught = false;
    try{
        iota.at(N);
    }catch(std::out_of_range&){
        caught = true;
    }
    assert(caught);

    // fill
    vtype one = zero<vtype>();
    ++one;
    atype aone;
    aone.fill(one);
    for(size_t i=0; i<N; ++i){
        assert(aone[i] == one);
    }

    // swap
    aone.swap(z);
    for(size_t i=0; i<N; ++i){
        assert(aone[i] == zero<vtype>());
        assert(z[i] == one);
    }

    // seed
    dummySeedSeq seedseq;
    aone = atype::seed(seedseq);
    vector<uint32_t> v32( N*((W+31)/32) );
    seedseq.generate(v32.begin(), v32.end());
    size_t jj=0;
    uint32_t mask = 0xffffffff;
    if( W < 32 )
        mask >>= (32-W);
    for(size_t i=0; i<N; ++i){
        for(size_t j=0; j<W; j+=32){
            uint32_t aj = get32(aone[i], j/32);
            assert( aj == (mask&v32.at(jj)) );
            jj++;
        }
    }

    // incr 

#ifndef __PGI
    atype a = {{}};
#else
    atype a;
    a.fill(zero<vtype>());
#endif
    a.incr();
    a.incr();
    a.incr();
    a.incr();
    assert( ull(a[0]) == 4u );
    assert( N<2 || ull(a[1]) == 0u );

    a.incr(0xbadcafe);
    assert( ull(a[0]) == (mask&(4u+0xbadcafe)) );

    // Set the zero'th entry to fff and then increment
    a[0] = fff<vtype>();
    a.incr();
    assert( a[0] == zero<vtype>() );
    assert( N<2 || ull(a[1]) == (W>8?1u:0xcc));

    a.incr();
    assert( ull(a[0]) == 1u );
    assert( N<2 || ull(a[1]) == (W>8?1u:0xcc) );

    R123_ULONG_LONG ulfff = ull(fff<vtype>());

    a.incr(ulfff);
    a.incr(ulfff - 5u);
    a.incr(2);
    a.incr(2);
    a.incr(2);
    a.incr(2);

    // operator<< and operator>>

    std::stringstream ss;
    ss << a;
    AType b;
    ss >> b;
    assert(a == b);
    cout << " OK\n";
}


int main(int, char **){
#if R123_USE_SSE
    doit<r123array1xm128i>(1, 128);
#endif
    doit<r123array2x32>(2, 32);
    doit<r123array4x32>(4, 32);
#if R123_USE_64BIT
    doit<r123array2x64>(2, 64);
    doit<r123array4x64>(4, 64);
#endif
    doit<r123array16x8>(16, 8);
    return 0;
}

