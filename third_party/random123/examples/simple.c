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
#include <Random123/threefry.h>
#include <stdio.h>
#include "example_seeds.h"

int main(int argc, char **argv){
    int i;
    uint32_t seed = example_seed_u32(EXAMPLE_SEED1_U32); // example of user-settable seed
    
    /* while this example starts the counter from 0 and then increments by 0,
       this could start anywhere and increment by any stride or pattern that
       makes sense for the application as long as it produces a non-repeating stream */
    threefry4x32_ctr_t  ctr = {{0,0,0,0}};
    /* we illustrate one user-specified seed and one constant as the key */
    threefry4x32_key_t key = {{seed, EXAMPLE_SEED2_U32,0,0}};
    (void)argc; (void)argv; /* unused */
    printf( "The first few randoms with key 0x%llx 0x%llx\n",
	   (unsigned long long)key.v[0], (unsigned long long)key.v[1]);
    for(i=0; i<10; ++i){
        ctr.v[0] = i;
	{
          threefry4x32_ctr_t rand = threefry4x32(ctr, key);
          printf("ctr: %llx %llx threefry4x32(20, ctr, key): %llx %llx\n",
                 (unsigned long long)ctr.v[0], (unsigned long long)ctr.v[1],
                 (unsigned long long)rand.v[0], (unsigned long long)rand.v[1]);
	}
    }
    return 0;
}
