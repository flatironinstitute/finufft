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
/*
 * This file is the Metal kernel.
 *
 * Written by Tom Schoonjans <Tom.Schoonjans@me.com>
 */

#include <Random123/threefry.h>

/*
 * counthits generates n x,y points and returns hits[tid] with
 * the count of number of those points within the unit circle on
 * each thread.
 */
kernel void counthits(const device unsigned  &n [[ buffer(0) ]],
                            device uint *hitsp [[ buffer(1) ]],
                            device uint *triesp [[ buffer(2) ]],
			           uint     tid [[thread_position_in_grid]]) {
    unsigned hits = 0, tries = 0;
    threefry4x32_key_t k = {{tid, 0xdecafbad, 0xfacebead, 0x12345678}};
    threefry4x32_ctr_t c = {{0, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
    const float uint_max_fl = (float) UINT_MAX;
    while (tries < n) {
	union {
	    threefry4x32_ctr_t c;
	    uint4 i;
	} u;
	c.v[0]++;
	u.c = threefry4x32(c, k);
	float x1 = ((float) u.i.x) / uint_max_fl, y1 = ((float) u.i.y) / uint_max_fl;
	float x2 = ((float) u.i.z) / uint_max_fl, y2 = ((float) u.i.w) / uint_max_fl;
	if ((x1*x1 + y1*y1) < (1.0)) {
	    hits++;
	}
	tries++;
	if ((x2*x2 + y2*y2) < (1.0)) {
	    hits++;
	}
	tries++;
    }
    hitsp[tid] = hits;
    triesp[tid] = tries;
}
