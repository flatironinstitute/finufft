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
#ifndef UTIL_M128_H__
#define UTIL_M128_H__
#include <Random123/features/sse.h>

// The formatting in fips-197 seems to correspond to
// byte[15] [14] ... [0]
__m128i m128i_from_charbuf(const char *s){
    unsigned int bytes[16];
    sscanf(s, "%02x%02x%02x%02x" "%02x%02x%02x%02x" "%02x%02x%02x%02x" "%02x%02x%02x%02x",
             &bytes[0], &bytes[1], &bytes[2], &bytes[3],
             &bytes[4], &bytes[5], &bytes[6], &bytes[7],
             &bytes[8], &bytes[9], &bytes[10], &bytes[11],
             &bytes[12], &bytes[13], &bytes[14], &bytes[15]);
    return _mm_set_epi8(
                         bytes[15], bytes[14], bytes[13], bytes[12],
                         bytes[11], bytes[10], bytes[9], bytes[8],
                         bytes[7], bytes[6], bytes[5], bytes[4],
                         bytes[3], bytes[2], bytes[1], bytes[0]
                         );
}

#define M128_STR_SIZE 34    /* minimum size of the charbuf "hex" argument */

char *m128i_to_charbuf(__m128i m, char *hex){
    union {
	unsigned char bytes[16];
	__m128i m;
    } u;
    _mm_storeu_si128((__m128i*)&u.bytes[0], m);
    sprintf(hex, "%02x%02x%02x%02x" "%02x%02x%02x%02x"
            " "
            "%02x%02x%02x%02x""%02x%02x%02x%02x",
             u.bytes[0], u.bytes[1], u.bytes[2], u.bytes[3],
             u.bytes[4], u.bytes[5], u.bytes[6], u.bytes[7],
             u.bytes[8], u.bytes[9], u.bytes[10], u.bytes[11],
             u.bytes[12], u.bytes[13], u.bytes[14], u.bytes[15]);

    return hex;
}

#ifdef __cplusplus
#include <string>

__m128i m128i_from_string(const std::string& s) {
    return m128i_from_charbuf(s.c_str());
}

std::string m128i_to_string(__m128i m) {
    char hex[M128_STR_SIZE];
    
    m128i_to_charbuf(m, hex);
    return std::string(hex);
}
#endif /* __cplusplus */

#endif /* UTIL_M128_H__ */
