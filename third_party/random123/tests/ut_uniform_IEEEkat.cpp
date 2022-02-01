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

 /* ut_uniform_IEEEkat.cpp - a "Known Answer Test" for uniform.hpp

    This code tests that the compilation environment reproduces
    exactly the behavior of u01, uneg11 and u01fixedpt on an x86-64
    system with strict IEEE arithmetic.  It is likely to fail on
    systems that use 80-bit internal registers (e.g., 32-bit x86), and
    systems that are smart enough to fuse floating point multiply and
    add into a single, rounded-only-once instruction (e.g., PowerPC,
    Fermi, newer ARMs, Haswell, Itanium, etc.).  Failures in these
    cases are *not* necessarily problematic. */

#include <Random123/uniform.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>

using namespace r123;

std::map<std::string, long double> katmap;

// Don't inline this.  It's called thousands of times in fill_katmap
// and blows out the optimizer in older versions of clang and open64
// if it's inlined.
void insert(const char *s, long double v){
    katmap[std::string(s)] = v;
}

void fill_katmap(){
#include "ut_uniform_IEEEkatvectors.hpp"
#if 0  // helpful for debug,
    for(std::map<std::string, long double>::iterator p=katmap.begin(); p!=katmap.end(); ++p){
        fprintf(stderr, "%s -> %La\n", p->first.c_str(), p->second);
    }
#endif
}

template <typename T>
typename r123::make_unsigned<T>::type U(T x){ return x; }

template <typename T>
typename r123::make_signed<T>::type S(T x){ return x; }
        
bool checking = true;
int nfail = 0;
int nuntested = 0;
int notfound = 0;

#define DO1(T, expr, astr) DoOne<T>(#expr, astr, expr)

#define DO(i, astr) do{                                     \
        ChkSignInvariance(#i, i);                      \
        if(std::numeric_limits<float>::digits == 24){ \
            DO1(float, u01<float>(i), astr);                  \
            DO1(float, uneg11<float>(i), astr);                 \
            DO1(float, u01fixedpt<float>(i), astr);              \
        }else{                                          \
            printf("UNTESTED: %s:  float does not have a 24 bit mantissa\n", #i); \
            nuntested++;                                                \
        }                                                       \
        if(std::numeric_limits<double>::digits == 53){ \
            DO1(double, u01<double>(i), astr);                 \
            DO1(double, uneg11<double>(i), astr);                \
            DO1(double, u01fixedpt<double>(i), astr);             \
        }else{                                          \
            printf("UNTESTED: %s:  double does not have a 53 bit mantissa\n", #i); \
            nuntested++;                                                \
        }                                                       \
        if(std::numeric_limits<long double>::digits == 64){     \
            DO1(long double, u01<long double>(i), astr);                \
            DO1(long double, uneg11<long double>(i), astr);              \
            DO1(long double, u01fixedpt<long double>(i), astr);          \
        }else{                                          \
            printf("UNTESTED: %s:  long double does not have a 64 bit mantissa\n", #i); \
            nuntested++;                                                \
        }                                                       \
    } while(0)

// u01, uneg11 and u01fixedpt should all depend on the bits, but not the
// signedness of their argument.  The templated functions S(i) and U(i)
// return their argument cast to an approprite signed and unsigned type.
// ChkSignInvariance verifies that.
template <typename IType>
void ChkSignInvariance(const std::string& s, IType i){
    if( u01<float>(S(i)) != u01<float>(U(i)) ){ 
        printf("INVARIANT FAILURE:  u01<float>(Signed(x)) != u01<float>(Unsigned(x)) x=%s\n", s.c_str()); 
        nfail++;                                                        
    }                                                                   
    if( uneg11<float>(S(i)) != uneg11<float>(U(i)) ){                   
        printf("INVARIANT FAILURE:  uneg11<float>(Signed(x)) != uneg11<float>(Unsigned(x)) x=%s\n", s.c_str());
        nfail++;                                                        
    }                                                                   
    if( u01fixedpt<float>(S(i)) != u01fixedpt<float>(U(i)) ){           
        printf("INVARIANT FAILURE:  u01<float>(Signed(x)) != u01<float>(Unsigned(x)) x=%s\n", s.c_str()); 
        nfail++;                                                        
    }                                                                   

    if( u01<double>(S(i)) != u01<double>(U(i)) ){ 
        printf("INVARIANT FAILURE:  u01<double>(Signed(x)) != u01<double>(Unsigned(x)) x=%s\n", s.c_str()); 
        nfail++;                                                        
    }                                                                   
    if( uneg11<double>(S(i)) != uneg11<double>(U(i)) ){                   
        printf("INVARIANT FAILURE:  uneg11<double>(Signed(x)) != uneg11<double>(Unsigned(x)) x=%s\n", s.c_str());
        nfail++;                                                        
    }                                                                   
    if( u01fixedpt<double>(S(i)) != u01fixedpt<double>(U(i)) ){           
        printf("INVARIANT FAILURE:  u01<double>(Signed(x)) != u01<double>(Unsigned(x)) x=%s\n", s.c_str()); 
        nfail++;                                                        
    }                                                                   

    if( u01<long double>(S(i)) != u01<long double>(U(i)) ){ 
        printf("INVARIANT FAILURE:  u01<long double>(Signed(x)) != u01<long double>(Unsigned(x)) x=%s\n", s.c_str()); 
        nfail++;                                                        
    }                                                                   
    if( uneg11<long double>(S(i)) != uneg11<long double>(U(i)) ){                   
        printf("INVARIANT FAILURE:  uneg11<long double>(Signed(x)) != uneg11<long double>(Unsigned(x)) x=%s\n", s.c_str());
        nfail++;                                                        
    }                                                                   
    if( u01fixedpt<long double>(S(i)) != u01fixedpt<long double>(U(i)) ){           
        printf("INVARIANT FAILURE:  u01<long double>(Signed(x)) != u01<long double>(Unsigned(x)) x=%s\n", s.c_str()); 
        nfail++;                                                        
    }                                                                   
}

template <typename T>
void DoOne(const std::string s, const char* astr, volatile T x){
    std::string ss = s + " a=" + astr;
    volatile long double ldx = x;
    if(checking){                                 
        if( katmap.find(ss) == katmap.end() ){   
            printf("NOT FOUND: katmap[%s]\n", ss.c_str());
            notfound++;                           
        }else{                                    
            if(ldx!=katmap[ss]){                  
                printf("MISMATCH:  %s: computed=%.21Lg reference=%.21Lg\n", ss.c_str(), ldx, katmap[ss]); 
                nfail++;                                                
            }                                                           
        }                                                               
    }else{                                                              
        printf("insert(\"%s\", %#.21LgL);\n", ss.c_str(), ldx);     
    }
}

void DO3264(int a){
    const uint32_t maxu32 = std::numeric_limits<uint32_t>::max();
    const uint64_t maxu64 = std::numeric_limits<uint64_t>::max();
    const uint32_t minu32 = std::numeric_limits<uint32_t>::min();
    const uint64_t minu64 = std::numeric_limits<uint64_t>::min();

    const int32_t maxi32 = std::numeric_limits<int32_t>::max();
    const int64_t maxi64 = std::numeric_limits<int64_t>::max();
    const int32_t mini32 = std::numeric_limits<int32_t>::min();
    const int64_t mini64 = std::numeric_limits<int64_t>::min();

    char astr[32];
    sprintf(astr, "%d", a);

    DO( minu32 + uint32_t(a), astr );
    DO( minu64 + uint64_t(a), astr );
    DO( mini32 + int32_t(a), astr ); 
    DO( mini64 + int64_t(a), astr ); 

    DO( maxu32 - uint32_t(a), astr );
    DO( maxu64 - uint64_t(a), astr );
    DO( maxi32 - int32_t(a), astr ); 
    DO( maxi64 - int64_t(a), astr ); 
}

int main(int argc, char **argv){
    if(argc>1){
        checking = false;
        printf("/* This file was created by '%s %s' on a reference\n"
               "   platform, and is #included in the recompilation of %s\n"
               "   on a target platform.  When %s is run with no arguments\n"
               "   on the target platform, it asserts that the values computed\n"
               "   on the target platform match the reference values recorded here.\n"
               "   These reference values were computed on an x86_64 using 32-bit,\n"
               "   64-bit and 80-bit IEEE arithmetic for float, double and long double\n"
               "   respectively.  Other platforms with different representations of\n"
               "   floating point values or different conventions for how intermediates\n"
               "   are stored and rounded will almost certainly fail these tests\n"
               "   even though their results might be perfectly valid.\n"
               "*/\n", argv[0], argv[1], argv[0], argv[0]);
    }
    fill_katmap();
    

    DO3264(0);
    DO3264(1);
    DO3264(2);
    DO3264(3);
    DO3264(4);
    DO3264(5);

    DO3264(63);
    DO3264(64);
    DO3264(65);

    DO3264(127);
    DO3264(128);
    DO3264(129);

    DO3264(191);
    DO3264(192);
    DO3264(193);

    DO3264(255);
    DO3264(256);
    DO3264(257);

    DO3264(319);
    DO3264(320);
    DO3264(321);

    DO3264(382);
    DO3264(383);
    DO3264(384);

    DO3264(639);
    DO3264(640);
    DO3264(641);

    DO3264(1023);
    DO3264(1024);
    DO3264(1025);

    DO3264(3070);
    DO3264(3071);
    DO3264(3072);

    DO3264(5119);
    DO3264(5120);
    DO3264(5121);

    if(notfound){
        printf("// %s: WARNING:  %d tests were not checked because reference values were not compiled in\n",
               argv[0], notfound);
    }
    if(nuntested){
        printf("// %s: WARNING:  %d tests were not performed because the floating point rep does not match the IEEE format used to compute reference values\n", 
               argv[0], nuntested);
    }
    if(nfail){
        printf("// %s: FAILED %d Known-Answer-Tests failed\n", argv[0], nfail);
        printf("Such failures may be due to non-IEEE arithmetic on your platform.  In some \n"
               "cases, you may be able to recover IEEE arithmetic by pre-defining the\n"
               "pp-symbol R123_UNIFORM_FLOAT_STORE to a non-zero value, e.g., adding\n"
               "-DR123_UNIFORM_FLOAT_STORE=1 to the compile command line.  On some\n"
               "systems (notably, 32-bit x86 architectures) this will prevent use of\n"
               "extra-wide internal floating point registers and will recover IEEE\n"
               "arithmetic.  Unfortunately, this will make u01 and uneg11 significantly\n"
               "slower, so you may not wish to define it in production code.  As far\n"
               "as we know, the floating point values returned with the symbol unset\n"
               "are perfectly reasonable.  They simply don't perfectly match the\n"
               "values computed on our reference x86-64 platform with IEEE arithmetic\n");
    }else{
        printf("// %s: SUCCESS\n", argv[0]);
    }

    return !!nfail;
}
