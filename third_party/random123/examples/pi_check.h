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
#ifndef PI_CHECK_H__
#define PI_CHECK_H__ 1

#include <stdio.h>

const unsigned long NTRIES = 10000000UL;

/* XX Cannot make this static, is included in some files that only use it
   under ifdef-conditionally, and we do not want to ifdef this to match. */
int pi_check(unsigned long hits, unsigned long tries)
{
    const double PI = 3.14159265358979323846;
    double ourpi, mean, var, delta, chisq;
    printf("%lu out of %lu darts thrown at a square board hit the inscribed circle\n",
           hits, tries);
    ourpi = 4.*hits/tries;
    printf("pi is approximately %.8g (diff = %.2g %%)\n", ourpi, (ourpi - PI)*100./PI);
    mean = tries*(PI/4.);
    var = tries * (PI/4.)*(1. - (PI/4.));
    delta = hits - mean;
    chisq = delta*delta/var;
    /*  Sigh.  Jump through hoops so we don't want to link with -lm for sqrt */
    if( chisq < 1. )
        printf("OK, # of hits is less than one 'sigma' away from expectation\n(chisquared = %.2g)\n", chisq);
    else if(chisq < 4.)
        printf("OK, # of hits is between one and two 'sigma' away from expectation\n(chisquared = %.2g)\n", chisq);
    else if(chisq < 9.)
        printf("Maybe OK, # of hits is between two and three 'sigma' away from expectation\n(chisquared = %.2g)\n", chisq);
    else {
        printf("May not be OK, # of hits is more than three 'sigma'.  Worth looking into.\n(chisquared = %.2g)\n", chisq);
	return 1;
    }
    return 0;
}

#endif /* PI_CHECK_H__ */
