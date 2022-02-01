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
// TODO - really do a thorough and complete set of tests.

#ifdef _MSC_FULL_VER
// Engines have multiple copy constructors, quite legal C++, disable MSVC complaint
#pragma warning (disable : 4521)
#endif

#include <Random123/philox.h>
#include <Random123/aes.h>
#include <Random123/threefry.h>
#include <Random123/ars.h>
#include <Random123/conventional/Engine.hpp>
#include <Random123/ReinterpretCtr.hpp>
#if R123_USE_CXX11_RANDOM
#include <random>
#endif
#include <cassert>
#include <iostream>
#include <sstream>
#include "util_demangle.hpp"

using namespace std;
using namespace r123;

template <typename EType>
typename EType::result_type kat1000(){
    // A zero return says that no KAT is known.  This makes
    // sense for the ReniterpretCtr-based engines which are
    // expected to produce endian-specific results, so we
    // don't have known answers for them.
    return 0;
}

#if R123_USE_64BIT
#if R123_USE_PHILOX_64BIT
template <> uint64_t kat1000<Engine<Philox2x64 > >(){ return R123_64BIT(10575809911605703474); }
#endif
template <> uint64_t kat1000<Engine<Threefry2x64 > >(){ return R123_64BIT(17578122881062615727); }
#endif
template <> uint32_t kat1000<Engine<Philox4x32 > >(){ return 1721865298; }
template <> uint32_t kat1000<Engine<Threefry4x32 > >(){ return 874101813; }
#if R123_USE_AES_OPENSSL
template <> uint8_t  kat1000<Engine<AESOpenSSL16x8> >(){ return 0237; }
#endif

#define ASSERTEQ(A, B) assert(A==B); assert(A()==B()); assert(A()==B()); assert(A()==B())

struct DummySeedSeq{
    template <typename ITER>
    void generate(ITER b, ITER e){
        std::fill(b, e, 1);
    }
};

template <typename EType>
void doit(){
    EType e;
    cout << "doit<" << demangle(e) << ">";
    typedef typename EType::cbrng_type BType;
    typedef typename EType::result_type rtype;
    typedef typename BType::ctr_type ctype;
    typedef typename BType::key_type ktype;

    DummySeedSeq dummyss;
    EType ess(dummyss);
    assert(ess != e);

    rtype r1 = e();
    rtype r2 = e(); assert( r1 != r2 );
    rtype r3 = e(); assert( r3 != r2 && r3 != r1 );

    // We've elsewhere confirmed that the underlying bijections actually "work",
    // e.g., that they pass the Known Answer Test for some set of test vectors.
    // Here, we simply check that the output of the Engine corresponds, in the expected
    // way to the output of the underlying bijection.
    
    // Check that the first few values out of the engine correspond
    // to output from the underlying bijection.
    BType b;
    ctype c1 = {{}};
    ktype k = e.getkey();
    e.seed(); 
    for(int i=0; i<100; ++i){
        c1[0]++;
        ctype rb = b(c1, k);
        for(typename ctype::reverse_iterator p=rb.rbegin(); p!=rb.rend(); ++p){
            rtype re = e();
            assert( *p == re );
        }
    }

    // Check that discard work as expected, i.e., we can keep two
    // engines "in sync" by discarding from one and stepping the other.
    EType e2;
    assert(e2 != e);
    e2.discard(100*c1.size());
    ASSERTEQ(e2, e);
    
    for(int disc=1; disc<50; ++disc){
        e.discard(disc);
        assert( e != e2 );
        for(int j=0; j<disc; ++j) e2();
        ASSERTEQ(e2, e);
    }

    // Check that saving and restoring state and the copy constructor 
    // works as expected.
    ostringstream oss;
    oss << e2;
    string s2 = oss.str();
    int fiftyfive = 55;
#if R123_USE_CXX11_TYPE_TRAITS
    // With CXX11, the library has type_traits to prevent
    // undesirable type resolution against the templated SeedSeq constructor
    EType e3(fiftyfive);
#else
    // Without CXX11, we have to be careful to pass in
    // a bona fide rtype, and not just something that will promote
    // to an rtype, if we want the rtype constructor.
    EType e3((rtype(fiftyfive)));
#endif
    EType esave(e);
    assert(e3 != e2);
    {
        istringstream iss(s2);
        iss >> e3;
    }
    ASSERTEQ(e3, e2);
    assert(e3 != esave );
    {
        istringstream iss(s2);
        iss >> e3;
    }
    ASSERTEQ(e3, esave);
    
    // Check that the constructor-from-rvalue works.
    EType e4((rtype)99);
    EType e5;
    assert(e4 != e5);
    assert(e4 != e3);
    e5.seed((rtype)99);
    ASSERTEQ(e4, e5);

#if R123_USE_STD_RANDOM
    // Check that we can use an EType with a std::distribution.
    // Obviously, this requires <random>
    uniform_int_distribution<int> dieroller(1, 6);
    vector<int> hist(7);
    int NROLL = 10000;
    for(int i=0; i<NROLL; ++i){
        int roll = dieroller(e5);
        hist[roll]++;
    }
    double chisq = 0.;
    double expected = NROLL/6.;
    double var = NROLL*5./36.;
    for(int pips=1; pips<=6; ++pips){
        double delta = hist[pips] - expected;
        chisq += delta*delta/var;
    }
    // The critical value of chisq with 6 degrees of freedom
    // for P=0.01 is 16.81.  For P=0.05, it is 12.59
    const double chicrit = 12.59;
    if( chisq > chicrit ){
        printf("std::uniform_int_distribution doesn't look random.  Chisq = %g.  Does this look like the result of a fair set of dice rolls to you?\n", chisq);
        for(int pips=1; pips<=6; ++pips){
            printf("%d pips  %d times\n", pips, hist[pips]);
        }
        abort();  // a bit harsh, no?  It might just be a rare event at the 5% level...
    }
#endif

    // Finally do a kat test.  
    EType ekat;
    ekat.discard(1000);
    typename EType::result_type r = ekat();
    typename EType::result_type knownanswer = kat1000<EType>();
    if( knownanswer != 0 && r != knownanswer )
        cerr << "KAT mismatch.  The 1000th random from " << demangle(ekat) << " is " << r << " it should be " << knownanswer << "\n";
    assert( knownanswer==0 || r == knownanswer );
    cout << " OK" << endl;
}

int main(int, char **){
#if R123_USE_PHILOX_64BIT
    doit<Engine<Philox2x64 > >();
    doit<Engine<ReinterpretCtr<r123array4x32, Philox2x64 > > >();
#endif
    doit<Engine<Philox4x32 > >();
    doit<Engine<Threefry4x32 > >();
#if R123_USE_64BIT
    doit<Engine<ReinterpretCtr<r123array4x32, Threefry2x64 > > >();
    doit<Engine<Threefry2x64 > >();
    doit<Engine<ReinterpretCtr<r123array2x64, Threefry4x32 > > >();
#endif

#if R123_USE_AES_NI
    if( haveAESNI() ){
        doit<Engine<ARS4x32> >();
#if R123_USE_64BIT
        doit<Engine<ReinterpretCtr<r123array2x64, ARS4x32> > >();
#endif
        doit<Engine<AESNI4x32> >();
    }else{
        cout << "AES is compiled into the binary, but is not available on this hardware\n";
    }
#endif
#if R123_USE_AES_OPENSSL
    doit<Engine<AESOpenSSL16x8> >();
#endif

    cout << "ut_Engine:  all OK" << endl;
    return 0;
}

