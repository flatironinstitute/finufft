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

#include "util.h"

#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/aes.h>
#include <Random123/ars.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>
#if R123_USE_X86INTRIN_H
#include <x86intrin.h>
#endif
#include "util_demangle.hpp"
#include "util.h"

using namespace r123;

const char *progname;
int debug = 0;
double cpu_hz = -1.0;

using namespace std;

namespace{
    template <typename B> void timer();
} // namespace <anon>

int main(int argc, char **argv){
    progname = argv[0];

    const char *envp;
    if( (envp = getenv("TIMERS_CPU_GHZ"))){
        cpu_hz = 1.e9 * atof(envp);
    }
#if R123_USE_AES_NI
    if( argc == 1 || strcmp(argv[1], "ARS")==0 ){
    if(haveAESNI()){
        timer<ARS1xm128i_R<5> >();
        timer<ARS4x32_R<5> >();
        timer<ARS1xm128i_R<7> >();
        timer<ARS4x32_R<7> >();
        timer<ARS1xm128i_R<10> >();
        timer<AESNI1xm128i >();
    }else{
        cout << "Skipping Bijections that use AES-NI instructions that are not available on this platform\n";
    } 
    }
#else
    cout << "This binary is not compiled with AES-NI support.  Skipping the ARS bijections\n";
#endif // R123_USE_AES_NI

    if( argc == 1 || strcmp(argv[1], "AES")==0 ){
#if R123_USE_AES_OPENSSL
    cout << "\n";
    timer<AESOpenSSL16x8>();
#endif
    }

    if( argc == 1 || strcmp(argv[1], "Threefry4x32")==0 ) {
    cout << "\n";
    timer<Threefry4x32_R<12> >();
    timer<Threefry4x32_R<20> >();
    timer<Threefry4x32 >();
    }

#if R123_USE_64BIT
    if( argc == 1 || strcmp(argv[1], "Threefry2x64")==0 ){
    cout << "\n";
    timer<Threefry2x64_R<13> >();
    timer<Threefry2x64_R<20> >();
    timer<Threefry2x64 >();
    }

    if( argc == 1 || strcmp(argv[1], "Threefry4x64")==0 ){
    cout << "\n";
    timer<Threefry4x64_R<12> >();
    timer<Threefry4x64_R<20> >();
    timer<Threefry4x64 >();
    timer<Threefry4x64_R<72> >();
    }
#else
    cout << "\n64bit types not implemented.  Skipping Threefry-Nx64 bijections\n";
#endif

#if R123_USE_PHILOX_64BIT
    if( argc == 1 || strcmp(argv[1], "Philox2x64") == 0 ){
    cout << "\n";
    timer<Philox2x64_R<6> >();
    timer<Philox2x64_R<10> >();
    }

    if( argc == 1 || strcmp(argv[1], "Philox4x64") == 0 ){
    cout << "\n";
    timer<Philox4x64_R<7> >();
    timer<Philox4x64_R<10> >();
    }
#else
    cout << "\n64x64->128bit multiplication is not implmented.  Skipping Philox-Nx64 bijections\n";
#endif

    if( argc == 1 || strcmp(argv[1], "Philox4x32") == 0 ){
    cout << "\n";
    timer<Philox4x32_R<7> >();
    timer<Philox4x32_R<10> >();
    }

    return 0;
}

    
namespace{

// To prevent the compiler from noticing that the result of the
// bijection is never used and eliding the entire calculation, we
// accumulate the output of millions of calls to the bijection.  All
// the ctr_types are sufficiently container-like that we can just
// loop over the contents, doing += on each value_type
template <typename CtrType>
CtrType& operator+=(CtrType& lhs, CtrType rhs){ 
    typename CtrType::const_iterator rp = rhs.cbegin();
    for(typename CtrType::iterator lp=lhs.begin(); lp!=lhs.end(); ++lp)
            *lp ^= *rp++;
        return lhs; 
}

// We've accumulated it, but we still have to use it.  A non-zero
// test serves the purpose:
template <typename CtrType>
bool nz(const CtrType v){
    for(typename CtrType::const_iterator vp=v.cbegin(); vp!=v.cend(); ++vp)
        if( *vp ) return true;
    return false;
}

#if R123_USE_AES_NI
// The "obvious" solution for ctr_types whose value_type doesn't
// have += defined (e.g., m128i) would be to overload += on the
// value_type.  But we can't do that in gcc because m128i is typedefed
// to a fancy compiler-specific builtin type, and you can only overload
// += on classes and enums.  So instead we specialize += on the
// array instead of on the value_type:
template<> 
r123array1xm128i& operator+=(r123array1xm128i& lhs, r123array1xm128i rhs){
    typedef r123array1xm128i CtrType;
    CtrType::const_iterator rp = rhs.cbegin();
    for(CtrType::iterator lp=lhs.begin(); lp!=lhs.end(); ++lp)
        lp->m = _mm_xor_si128(*lp, *rp++);
    return lhs;
}
#endif

template <typename B>
void timer(){
    typedef typename B::ctr_type ctr_type;
    ctr_type sum = {{}};
    uint_fast64_t N = 1000000;    // First try only a few thousand...
    B b;
    int bytes_per_call = sizeof(ctr_type);
    cout << demangle(b) << ": gran: " << bytes_per_call;

    ctr_type c0 = {{}};
    const char *envp;
    if((envp = getenv("TIMERS_COUNTER"))){
        std::istringstream iss((std::string(envp)));
        iss >> c0;
    }
     
    typename B::ukey_type uk = {{}};
    if( (envp = getenv("TIMERS_KEY"))){
        std::istringstream iss((std::string(envp)));
        iss >> uk;
    }
    typename B::key_type k(uk);
   
    ctr_type c = c0;
    double clk;
    ::timer(&clk);
    for(uint_fast64_t i=0; i<N; ++i){
        c.incr();
        sum += b(c, k);
    }
    double dur = ::timer(&clk);

    double bestrate = 0.;
    double bestdur = 0.;
    uint_fast64_t bestN = 0;
    for(size_t t=0; t<5; ++t){
        ctr_type c = c0;
        N = (uint_fast64_t)(N*(0.1/dur));
        ::timer(&clk);
        for(uint_fast64_t i=0; i<N; ++i){
            c.incr();
            sum += b(c, k);
        }
        dur = ::timer(&clk);
        double rate = N*bytes_per_call/dur;
        if( rate > bestrate ){
            bestrate = rate;
            bestN = N;
            bestdur = dur;
        }
    }
    cout << " (best of 5) " << bestN << " bijections in " << bestdur << " sec. rate: " << bestrate*1.e-9 << "GB/s";
    if( cpu_hz > 0. )
        cout << " cpB: " << cpu_hz/bestrate;
    cout << endl;
        
    if(!nz(sum))
        cout << "Don't let the compiler optimize it all away... sum==0.  That's a surprise!\n";
}

} // namespace <anonymous>


