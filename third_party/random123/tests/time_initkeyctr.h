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
#ifndef TIME_INITKEYCTR_H__
#define TIME_INITKEYCTR_H__ 1

/*
 * EXAMPLE_KEY* and EXAMPLE_CTR* values are just arbitrary numbers
 * with some bits set, they have no special importance, they do
 * not even have to be different.  If they are changed, then the
 * good_* values further below will need to be updated to match.
 */

#define EXAMPLE_KEY0 0xdeadbeefU
#define EXAMPLE_KEY1 0x12345678U
#define EXAMPLE_KEY2 0xc0debad1U
#define EXAMPLE_KEY3 0x31415926U

#define EXAMPLE_CTR0 0x00000000U
#define EXAMPLE_CTR1 0x10000000U
#define EXAMPLE_CTR2 0x20000000U
#define EXAMPLE_CTR3 0x30000000U

/*
 * The magic hex numbers below are just known good values that
 * result from the arbitrary EXAMPLE* inputs above.  We check that
 * we got these results at the end of timing tests to ensure that
 * we didn't accidentally let a compiler optimize away some loop.
 */
#if R123_USE_PHILOX_64BIT
static philox2x64_ctr_t good_philox2x64_6 = {{R123_64BIT(0xdd40cdb81af968d2),R123_64BIT(0x0cb57d6d5f7b68dc)}};
static philox2x64_ctr_t good_philox2x64_10 = {{R123_64BIT(0x539e5b3d18faf5da),R123_64BIT(0x838ca1328d07d3ba)}};
static philox4x64_ctr_t good_philox4x64_7 = {{R123_64BIT(0xcf492074862957a2),R123_64BIT(0x7057627260938584),R123_64BIT(0x676e23214a14901d),R123_64BIT(0xefa2c5df3848e3fe)}};
static philox4x64_ctr_t good_philox4x64_10 = {{R123_64BIT(0x1b64f56b381a5a89),R123_64BIT(0x940a282a8add45e1),R123_64BIT(0x53c936376ac7d5df),R123_64BIT(0x6147e87ec9bd9caa)}};
#endif
static philox4x32_ctr_t good_philox4x32_7 = {{0x40ba6a95,0x799e6a43,0x7dcabe10,0xa7a81636}};
static philox4x32_ctr_t good_philox4x32_10 = {{0xf16d828e,0xa1c5962d,0xacac820c,0x58113d7a}};
static threefry4x32_ctr_t good_threefry4x32_12 = {{0xe461db1c,0xfdfa62a7,0x0b10cd2a,0xa3679758}};
static threefry4x32_ctr_t good_threefry4x32_20 = {{0xf82cf576,0x162ca116,0x3afefe23,0x54cc64ac}};
#if R123_USE_64BIT
static threefry2x64_ctr_t good_threefry2x64_13 = {{R123_64BIT(0xdf0f096c179ad798),R123_64BIT(0x077862fbaa1a0d11)}};
static threefry2x64_ctr_t good_threefry2x64_20 = {{R123_64BIT(0xb91153d59815d50e),R123_64BIT(0xdb0dd45e5b0eab81)}};
static threefry4x64_ctr_t good_threefry4x64_12 = {{R123_64BIT(0x416d1802da0a4a0f),R123_64BIT(0xabd4d80749306281),R123_64BIT(0x62c6b120b542bff0),R123_64BIT(0xefb28dc80c6fc36c)}};
static threefry4x64_ctr_t good_threefry4x64_20 = {{R123_64BIT(0xad8f0b8c18ed5187),R123_64BIT(0xd80146a6961e1880),R123_64BIT(0x7fce9d950d8acbc4),R123_64BIT(0x782948d5203519f1)}};
static threefry4x64_ctr_t good_threefry4x64_72 = {{R123_64BIT(0x73ff3f7a0b878f68),R123_64BIT(0x6668f6bbaba83f31),R123_64BIT(0x088eb85d40fbdb56),R123_64BIT(0xd1f39136adc96552)}};
#endif

#if R123_USE_AES_NI
static ars4x32_ctr_t good_ars4x32_5 = {{0x279f6b0b, 0xd0b1edf6, 0x6044b433, 0x66c06817}};
static ars4x32_ctr_t good_ars4x32_7 = {{0xa9cd8055, 0x80272a47, 0x4b7ab914, 0x5351d78e}};
static aesni4x32_ctr_t good_aesni4x32_10 = {{0x1e68c9fd, 0x347b0858, 0x503d8d91, 0x9e73460a}};
#endif

/*
 * template code initializes a ukey and counter to known values
 * with a known offset and calls a Random123 test function with
 * that ukey, ctr and a count and closure.  keyctroffset is
 * a variable initialized from runtime environment (e.g. argv, argc,
 * getenv(), etc) to avoid compile-time optimization
 * caused by constants, so we get worst-case numbers.  Users may,
 * of course, benefit from compile-time optimization if they
 * have some constants for key or ctr values.
 * 
 */
#define TEST_TPL(NAME, N, W, R) \
if ((strncmp(#NAME, "aes", 3) == 0 || strncmp(#NAME, "ars", 3) == 0) && !haveAESNI()) { \
    printf("AESNI not available on this hardware\n"); \
} else { \
    NAME##N##x##W##_ukey_t ukey={{0}};            \
    NAME##N##x##W##_ctr_t ctr={{0}};            \
    size_t xi; \
    for (xi = 0; xi < sizeof(ukey)/sizeof(ukey.v[0]); xi++) { \
	switch (xi) { \
	    case 0: ukey.v[xi] = EXAMPLE_KEY0+keyctroffset; break; \
	    case 1: ukey.v[xi] = EXAMPLE_KEY1+keyctroffset; break; \
	    case 2: ukey.v[xi] = EXAMPLE_KEY2+keyctroffset; break; \
	    case 3: ukey.v[xi] = EXAMPLE_KEY3+keyctroffset; break; \
	} \
    } \
    for (xi = 0; xi < N; xi++) { \
	switch (xi) { \
	    case 0: ctr.v[xi] = EXAMPLE_CTR0+keyctroffset; break; \
	    case 1: ctr.v[xi] = EXAMPLE_CTR1+keyctroffset; break; \
	    case 2: ctr.v[xi] = EXAMPLE_CTR2+keyctroffset; break; \
	    case 3: ctr.v[xi] = EXAMPLE_CTR3+keyctroffset; break; \
	} \
    } \
    NAME##N##x##W##_##R(ctr, ukey, good_##NAME##N##x##W##_##R, count, infop); \
}

#include "util_expandtpl.h"

#endif /* TIME_INITKEYCTR_H__ */


