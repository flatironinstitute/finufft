/*
Copyright 2013, D. E. Shaw Research.
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

 /* ut_uniform.cpp:   unit test for uniform.hpp.  

    This is a "sanity test" of u01, uneg11 and u01fixedpt.  We confirm
    that a histogram of few thousand calls to each of the functions
    matches a reference histogram.  This verifies that the results are
    generally sane i.e., they fall within the expected range, and that
    they are close to a correct distribution.  It is *not* a foolproof
    test of correctness, but it should catch portability issues
    like errors in r123::make_signed or r123::make_unsigned
    or r123::maxTvalue or misunderstandings about std::numeric_limits.

    There is a "known answer test" for uniform.hpp in ut_uniform_IEEEkat.cpp,
    but it is only expected to work on machines with strict IEEE arithmetic
    and  no high-precicision intermediates.  See its own comments for
    more details.
 */

#include <Random123/uniform.hpp>
#include <Random123/threefry.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <sstream>

using namespace r123;

template <typename T>
typename r123::make_unsigned<T>::type U(T x){ return x; }

template <typename T>
typename r123::make_signed<T>::type S(T x){ return x; }
        
#define Chk(u, Rng, Ftype) do{                            \
        chk<Ftype, Rng>(#u, #Rng, #Ftype, &u<Ftype, Rng::ctr_type::value_type>); \
    }while(0)

std::map<std::string, std::string> refmap;

void RefHist(const char* k, const char *v){
    refmap[std::string(k)] = std::string(v);
}

void fillrefhist(){
#include "ut_uniform_reference.hpp"
}

bool checking = true;
int nfail = 0;

template<typename Ftype, typename RNG, typename Utype>
void chk(const std::string& fname, const std::string& rngname, const std::string& ftypename, Utype f){
    std::string key = fname + " " + rngname + " " + ftypename;
    RNG rng;
    typedef typename RNG::ukey_type ukey_type;
    typedef typename RNG::ctr_type ctr_type;
    typedef typename RNG::key_type key_type;

    ctr_type c = {{}};
    ukey_type uk = {{}};
    key_type k = uk;
    // 26 bins - 13 greater than 0 and 13 less.  Why 13?  Because a
    // prime number seems less likely to tickle the rounding-related
    // corner cases, which is aruably both good and bad.
    const int NBINS=26;
    
    int hist[NBINS] = {};
    for(int i=0; i<1000; ++i){
        c = c.incr();
        ctr_type r = rng(c, k);
        for(int j=0; j<ctr_type::static_size; ++j){
            Ftype u = f(r[j]);
            //printf("%s %llx, %.17g\n", key.c_str(), (long long)r[j], (double)u);
            R123_ASSERT( u >= -1.);
            R123_ASSERT( u <= 1.);
            int idx = (int) ((u + Ftype(1.))*Ftype(NBINS/2));
            hist[idx]++;
        }
    }
    std::ostringstream oss;
    for(int i=0; i<NBINS; ++i){
        oss << " " << hist[i];
    }
    if(checking){
        if( oss.str() != refmap[key] ){
            printf("MISMATCH:  %s:\n\tcomputed histogram=%s\n\treference histogram=%s\n", 
                   key.c_str(),
                   oss.str().c_str(),
                   refmap[key].c_str());
            nfail++;
        }
    }else{
        printf("RefHist(\"%s\", \"%s\");\n",  key.c_str(), oss.str().c_str());
    }
}

int main(int argc, char **argv){
    checking = (argc==1);
    fillrefhist();

    // 18 tests:  3 functions (u01, uneg11, u01fixedpt)
    //          x 2 input sizes (32 bit or 64 bit)
    //          x 3 output sizes (float, double, long double)
    Chk(u01, Threefry4x32, float);
    Chk(u01, Threefry4x32, double);
    Chk(u01, Threefry4x32, long double);

#if R123_USE_64BIT
    Chk(u01, Threefry4x64, float);
    Chk(u01, Threefry4x64, double);
    Chk(u01, Threefry4x64, long double);
#endif

    Chk(uneg11, Threefry4x32, float);
    Chk(uneg11, Threefry4x32, double);
    Chk(uneg11, Threefry4x32, long double);

#if R123_USE_64BIT
    Chk(uneg11, Threefry4x64, float);
    Chk(uneg11, Threefry4x64, double);
    Chk(uneg11, Threefry4x64, long double);
#endif
    
    Chk(u01fixedpt, Threefry4x32, float);
    Chk(u01fixedpt, Threefry4x32, double);
    Chk(u01fixedpt, Threefry4x32, long double);

#if R123_USE_64BIT
    Chk(u01fixedpt, Threefry4x64, float);
    Chk(u01fixedpt, Threefry4x64, double);
    Chk(u01fixedpt, Threefry4x64, long double);
#endif

    if(nfail){
        printf("// %s: FAILED %d Known-Answer-Tests failed\n", argv[0], nfail);
    }else if(checking){
        printf("%s: SUCCESS\n", argv[0]);
    }

    return !!nfail;
}
