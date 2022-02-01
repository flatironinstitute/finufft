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
// This "unit test" is basically a test of the completeness
// of compilerfeatures.hpp.  Each of the pp-symbols in compilerfeatures.hpp
// is supposed to have a definition.  We check them all, and
// in some cases, emit some appropriate code to check that
// they reflect reality.
#include <assert.h>
#include <Random123/features/compilerfeatures.h>
#include <iostream>

struct Outputter{
    Outputter(const char *name, int value){
        std::cout << name << " " << value << std::endl;
    }
};

// Many symbols rely on the pp-convention of
// expanding undefined values in arithmetic expressions to 0.
// Thus, we can't do something terse like:
// #define Out(Sym) Outputter outputter##Sym(#Sym, Sym)
// Instead, we have to force the preprocessor to evaluate
// the symbol.
// #if Sym
// Otrue(Sym)
// #else
// Ofalse(Sym)
// #endif
#define Otrue(Sym) Outputter outputter##Sym(#Sym, true)
#define Ofalse(Sym) Outputter outputter##Sym(#Sym, false)

#ifndef R123_USE_X86INTRIN_H
#error "No  R123_USE_X86INTRIN_H"
#endif
#if R123_USE_X86INTRIN_H
#include <x86intrin.h>
Otrue(R123_USE_X86INTRIN_H);
#else
Ofalse(R123_USE_X86INTRIN_H);
#endif

#ifndef R123_USE_IA32INTRIN_H
#error "No  R123_USE_IA32INTRIN_H"
#endif
#if R123_USE_IA32INTRIN_H
Otrue(R123_USE_IA32INTRIN_H);
#include <ia32intrin.h>
#else
Ofalse(R123_USE_IA32INTRIN_H);
#endif

#ifndef R123_USE_XMMINTRIN_H
#error "No  R123_USE_XMMINTRIN_H"
#endif
#if R123_USE_XMMINTRIN_H
#include <xmmintrin.h>
Otrue(R123_USE_XMMINTRIN_H);
#else
Ofalse(R123_USE_XMMINTRIN_H);
#endif

#ifndef R123_USE_EMMINTRIN_H
#error "No  R123_USE_EMMINTRIN_H"
#endif
#if R123_USE_EMMINTRIN_H
#include <emmintrin.h>
Otrue(R123_USE_EMMINTRIN_H);
#else
Ofalse(R123_USE_EMMINTRIN_H);
#endif

#ifndef R123_USE_SMMINTRIN_H
#error "No  R123_USE_SMMINTRIN_H"
#endif
#if R123_USE_SMMINTRIN_H
Otrue(R123_USE_SMMINTRIN_H);
#include <smmintrin.h>
#else
Ofalse(R123_USE_SMMINTRIN_H);
#endif

#ifndef R123_USE_WMMINTRIN_H
#error "No  R123_USE_WMMINTRIN_H"
#endif
#if R123_USE_WMMINTRIN_H
Otrue(R123_USE_WMMINTRIN_H);
#include <wmmintrin.h>
#else
Ofalse(R123_USE_WMMINTRIN_H);
#endif

#ifndef R123_USE_INTRIN_H
#error "No  R123_USE_INTRIN_H"
#endif
#if R123_USE_INTRIN_H
Otrue(R123_USE_INTRIN_H);
#include <intrin.h>
#else
Ofalse(R123_USE_INTRIN_H);
#endif

#ifndef R123_USE_SSE
#error "No  R123_USE_SSE"
#endif
#if R123_USE_SSE
Otrue(R123_USE_SSE);
#include <Random123/features/sse.h>
__m128i mm;
#else
Ofalse(R123_USE_SSE);
#endif

#ifndef R123_CUDA_DEVICE
#error "No  R123_CUDA_DEVICE"
#endif
R123_CUDA_DEVICE void cuda_device_func(){}

// C++11 features
#ifndef R123_USE_CXX11_UNRESTRICTED_UNIONS
#error "No  R123_USE_CXX11_UNRESTRICTED_UNIONS"
#endif
#if R123_USE_CXX11_UNRESTRICTED_UNIONS
Otrue(R123_USE_CXX11_UNRESTRICTED_UNIONS);
struct defaulted_ctor{
    int i;
    defaulted_ctor()=default;
    defaulted_ctor(const defaulted_ctor& d) : i(d.i){}
};
union unrestricted{
    int i;
    defaulted_ctor dc;
};
#else
Ofalse(R123_USE_CXX11_UNRESTRICTED_UNIONS);
#endif

#ifndef R123_USE_CXX11_STATIC_ASSERT
#error "No  R123_USE_CXX11_STATIC_ASSERT"
#endif
#if R123_USE_CXX11_STATIC_ASSERT
Otrue(R123_USE_CXX11_STATIC_ASSERT);
static_assert(true, "this shouldn't be a problem");
#else
Ofalse(R123_USE_CXX11_STATIC_ASSERT);
#endif

#ifndef R123_USE_CXX11_CONSTEXPR
#error "No  R123_USE_CXX11_CONSTEXPR"
#endif
#if R123_USE_CXX11_CONSTEXPR
Otrue(R123_USE_CXX11_CONSTEXPR);
constexpr int zero() {return 0;}
#else
Ofalse(R123_USE_CXX11_CONSTEXPR);
#endif

#ifndef R123_USE_CXX11_EXPLICIT_CONVERSIONS
#error "No  R123_USE_CXX11_EXPLICIT_CONVERSIONS"
#endif
#if R123_USE_CXX11_EXPLICIT_CONVERSIONS
Otrue(R123_USE_CXX11_EXPLICIT_CONVERSIONS);
struct explicit_converter{
    explicit operator bool() const {return true;}
};
#else
Ofalse(R123_USE_CXX11_EXPLICIT_CONVERSIONS);
#endif

#ifndef R123_USE_CXX11_RANDOM
#error "No   R123_USE_CXX11_RANDOM"
#endif
#if R123_USE_CXX11_RANDOM
Otrue(R123_USE_CXX11_RANDOM);
#include <random>
#else
Ofalse(R123_USE_CXX11_RANDOM);
#endif

#ifndef R123_USE_CXX11_TYPE_TRAITS
#error "No  R123_USE_CXX11_TYPE_TRAITS"
#endif
#if R123_USE_CXX11_TYPE_TRAITS
Otrue(R123_USE_CXX11_TYPE_TRAITS);
#include <type_traits>
#else
Ofalse(R123_USE_CXX11_TYPE_TRAITS);
#endif

#ifndef R123_USE_CXX11_LONG_LONG
#error "No  R123_USE_CXX11_LONG_LONG"
#endif
#if R123_USE_CXX11_LONG_LONG
Otrue(R123_USE_CXX11_LONG_LONG);
unsigned long long ull;
#else
Ofalse(R123_USE_CXX11_LONG_LONG);
#endif

#ifndef R123_USE_CXX11_STD_ARRAY
#error "No  R123_USE_CXX11_STD_ARRAY"
#endif
#if R123_USE_CXX11_STD_ARRAY
Otrue(R123_USE_CXX11_STD_ARRAY);
#include <array>
std::array<int, 4> sai4;
#else
Ofalse(R123_USE_CXX11_STD_ARRAY);
#endif

#ifndef R123_FORCE_INLINE
#error "No  R123_FORCE_INLINE"
#endif
inline R123_FORCE_INLINE(int forcibly_inlined(int i));
inline int forcibly_inlined(int i){ return i+1;}

#ifndef R123_USE_AES_NI
#error "No  R123_USE_AES_NI"
#endif
#if R123_USE_AES_NI
Otrue(R123_USE_AES_NI);
__m128i aes(__m128i in){
    if( haveAESNI() )
        return _mm_aesenc_si128(in, in);
    else
        return _mm_setzero_si128();
}
#else
Ofalse(R123_USE_AES_NI);
#endif

#ifndef R123_USE_SSE4_2
#error "No  R123_USE_SSE4_2"
#endif
#if R123_USE_SSE4_2
Otrue(R123_USE_SSE4_2);
__m128i sse42(__m128i in){
    return _mm_cmpgt_epi64(in, in);
}
#else
Ofalse(R123_USE_SSE4_2);
#endif

#ifndef R123_USE_SSE4_1
#error "No  R123_USE_SSE4_1"
#endif
#if R123_USE_SSE4_1
Otrue(R123_USE_SSE4_1);
int sse41(__m128i in){
    return _mm_testz_si128(in, in);
}
#else
Ofalse(R123_USE_SSE4_1);
#endif

#ifndef R123_USE_AES_OPENSSL
#error "No  R123_USE_AES_OPENSSL"
#endif
#if R123_USE_AES_OPENSSL
Otrue(R123_USE_AES_OPENSSL);
#include <openssl/aes.h>
#else
Ofalse(R123_USE_AES_OPENSSL);
#endif

#ifndef R123_USE_GNU_UINT128
#error "No  R123_USE_GNU_UINT128"
#endif
#if R123_USE_GNU_UINT128
Otrue(R123_USE_GNU_UINT128);
__uint128_t u128;
#else
Ofalse(R123_USE_GNU_UINT128);
#endif

#ifndef R123_USE_ASM_GNU
#error "No  R123_USE_ASM_GNU"
#endif
#if R123_USE_ASM_GNU
Otrue(R123_USE_ASM_GNU);
#if defined(__x86_64__) || defined(__i386__)
int use_gnu_asm(){
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__ ("cpuid": "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx) :
                      "a" (1));
    return (ecx>>25) & 1;
}
#else
int use_gnu_asm(){ return 0; }
#endif
#else
Ofalse(R123_USE_ASM_GNU);
#endif

#ifndef R123_USE_CPUID_MSVC
#error "No  R123_USE_CPUID_MSVC"
#endif
#if R123_USE_CPUID_MSVC
Otrue(R123_USE_CPUID_MSVC);
int chkcpuid(){
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    return CPUInfo[2]&(1<<25);
}
#else
Ofalse(R123_USE_CPUID_MSVC);
#endif

#ifndef R123_USE_MULHILO32_ASM
#error "No  R123_USE_MULHILO32_ASM"
#endif
#if R123_USE_MULHILO32_ASM
Otrue(R123_USE_MULHILO32_ASM);
#else
Ofalse(R123_USE_MULHILO32_ASM);
#endif

#ifndef R123_USE_MULHILO64_ASM
#error "No  R123_USE_MULHILO64_ASM"
#endif
#if R123_USE_MULHILO64_ASM
Otrue(R123_USE_MULHILO64_ASM);
#else
Ofalse(R123_USE_MULHILO64_ASM);
#endif

#ifndef R123_USE_MULHILO64_MSVC_INTRIN
#error "No  R123_USE_MULHILO_MSVC_INTRIN"
#endif
#if R123_USE_MULHILO64_MSVC_INTRIN
Otrue(R123_USE_MULHILO64_MSVC_INTRIN);
#include <cstdint>
void msvc64mul(){
    uint64_t a=1000000000000000000;
    uint64_t b=a;
    uint64_t h, l;
    l = _umul128(a, b, &h);
    assert( l == a*b);
    assert( h == 54210108624275221ULL );
}
#else
Ofalse(R123_USE_MULHILO64_MSVC_INTRIN);
#endif

#ifndef R123_USE_MULHILO64_CUDA_INTRIN
#error "No  R123_USE_MULHILO64_CUDA_INTRIN"
#endif
#if R123_USE_MULHILO64_CUDA_INTRIN
Otrue(R123_USE_MULHILO64_CUDA_INTRIN);
#else
Ofalse(R123_USE_MULHILO64_CUDA_INTRIN);
#endif

#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
#error "No  R123_USE_MULHILO64_OPENCL_INTRIN"
#endif
#if R123_USE_MULHILO64_OPENCL_INTRIN
Otrue(R123_USE_MULHILO64_OPENCL_INTRIN);
#else
Ofalse(R123_USE_MULHILO64_OPENCL_INTRIN);
#endif

#ifndef R123_USE_MULHILO64_MULHI_INTRIN
#error "No  R123_USE_MULHILO64_MULHI_INTRIN"
#endif
#if R123_USE_MULHILO64_MULHI_INTRIN
Otrue(R123_USE_MULHILO64_MULHI_INTRIN);
static int test_mulhilo64_intrin(){
    uint64_t a = R123_64BIT(0x1234567887654321);
    uint64_t b = R123_64BIT(0x8765432112345678);
    uint64_t c = R123_MULHILO64_MULHI_INTRIN(a, b);
    assert( c == R123_64BIT(0x09A0CD05B99FE92E) );
    return c == R123_64BIT(0x09A0CD05B99FE92E);
}
int mulhilo64_intrin_ok = test_mulhilo64_intrin();
#else
Ofalse(R123_USE_MULHILO64_MULHI_INTRIN);
#endif

#ifndef R123_USE_MULHILO32_MULHI_INTRIN
#error "No  R123_USE_MULHILO32_MULHI_INTRIN"
#endif
#if R123_USE_MULHILO32_MULHI_INTRIN
Otrue(R123_USE_MULHILO32_MULHI_INTRIN);
static int test_mulhilo32_intrin(){
    uint64_t a32 = 0x12345678;
    uint64_t b32 = 0x87654321;
    uint64_t c32 = R123_MULHILO32_MULHI_INTRIN(a32, b32);
    assert( c32 == 0x09A0CD05 );
    return c32 == 0x09A0CD05;
}
int mulhilo32_intrin_ok = test_mulhilo32_intrin();
#else
Ofalse(R123_USE_MULHILO32_MULHI_INTRIN);
#endif

#ifndef R123_USE_MULHILO64_C99
#error "No  R123_USE_MULHILO64_C99"
#endif
#if R123_USE_MULHILO64_C99
Otrue(R123_USE_MULHILO64_C99);
#else
Ofalse(R123_USE_MULHILO64_C99);
#endif

#ifndef R123_64BIT
#error "No R123_64BIT"
#else
void xx() {
    uint64_t a = R123_64BIT(0x1234567890abcdef);
    assert ( (a >> 60) == 0x1 );
}
#endif

#ifndef R123_USE_PHILOX_64BIT
#error "No  R123_USE_PHILOX_64BIT"
#endif
#if R123_USE_PHILOX_64BIT
Otrue(R123_USE_PHILOX_64BIT);
#else
Ofalse(R123_USE_PHILOX_64BIT);
#endif

#ifndef R123_ASSERT
#error "No  R123_ASSERT"
#else
void chkassert(){
    R123_ASSERT(1);
}
#endif

#ifndef R123_STATIC_ASSERT
#error "No  R123_STATIC_ASSERT"
#else
R123_STATIC_ASSERT(1, "looks true to me");
void chkstaticassert(){
    R123_STATIC_ASSERT(1, "it's ok inside a function too");
}
#endif

int main(int , char **){return 0;}
