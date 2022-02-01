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
#include <Random123/ars.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if !R123_USE_SSE
int main(int argc, char **argv){
    (void)argc; (void)argv; /* unused */
    printf("No SSE support.  This test is not compiled\n");
    return 0;
}
#else
#include "util_m128.h"

int
main(int argc, char **argv)
{
#if R123_USE_AES_NI
    struct r123array1xm128i c, k, ret;
    char m128str[M128_STR_SIZE], *kat;

    if (haveAESNI()) {
	c.v[0].m = m128i_from_charbuf("01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00");
	k.v[0].m = m128i_from_charbuf("01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00");
	ret = ars1xm128i_R(7, c, k);
	kat = "2b1623350cd214dc 7740187993411872";
	if (strcmp(m128i_to_charbuf(ret.v[0].m, m128str), kat) != 0) {
	    fprintf(stderr, "%s: error, expected %s, got %s\n", argv[0], kat, m128str);
	    exit(1);
	}
	printf("%s: OK, got %s\n", argv[0], kat);
	c.v[0].m = m128i_from_charbuf("00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00");
	k.v[0].m = m128i_from_charbuf("01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00");
	ret = ars1xm128i_R(7, c, k);
	kat = "2de6b66fa461b668 f380126f32b9cd22";
	if (strcmp(m128i_to_charbuf(ret.v[0].m, m128str), kat) != 0) {
	    fprintf(stderr, "%s: error, expected %s, got %s\n", argv[0], kat, m128str);
	    exit(1);
	}
	printf("%s: OK, got %s\n", argv[0], kat);
    } else {
	printf("%s: no AES-NI on this machine\n", argv[0]);
    }
#else
    printf("%s: no AES-NI compiled into this program\n", argv[0]);
#endif
    (void)argc; (void)argv; /* unused */
    return 0;
}

#endif
