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
#ifndef TEST_TPL
#error "TEST_TPL not defined before including util_expandtpl.h"
#else
/*
 * This is included by various files after defining TEST_TPL to
 * expand TEST_TPL for each of the RNGs we want to test.
 * TEST_TPL args are the name of the RNG, N, W, and R,
 * N being the number of words, W being the wordsize in bits,
 * and R being the number of rounds.
 */

#if TRY_PHILOX2X32
TEST_TPL(philox, 2, 32, 7)
TEST_TPL(philox, 2, 32, 10)
#endif
TEST_TPL(philox, 4, 32, 7)
TEST_TPL(philox, 4, 32, 10)
#if R123_USE_PHILOX_64BIT
TEST_TPL(philox, 2, 64, 6)
TEST_TPL(philox, 2, 64, 10)
TEST_TPL(philox, 4, 64, 7)
TEST_TPL(philox, 4, 64, 10)
#endif
#if R123_USE_64BIT
TEST_TPL(threefry, 2, 64, 13)
TEST_TPL(threefry, 2, 64, 20)
TEST_TPL(threefry, 4, 64, 12)
TEST_TPL(threefry, 4, 64, 20)
TEST_TPL(threefry, 4, 64, 72)
#endif
TEST_TPL(threefry, 4, 32, 12)
TEST_TPL(threefry, 4, 32, 20)

#if R123_USE_AES_NI
TEST_TPL(ars, 4, 32, 5)
TEST_TPL(ars, 4, 32, 7)
TEST_TPL(aesni, 4, 32, 10)
#endif

#undef TEST_TPL
#endif /* TEST_TPL */
