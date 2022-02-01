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

#if __cplusplus<201103L
#include <iostream>
int main(int, char**){
    std::cout << "ua.hpp requires C++11.  No tests performed\n";
    return 0;
}
#else

#include <Random123/array.h>
#include <Random123/threefry.h>
#include <Random123/uniform.hpp>

using namespace r123;
int main(int, char **){
    Threefry4x32 rng;
    Threefry4x32::ctr_type c = {{1, 2, 3, 4}};
    Threefry4x32::ukey_type uk = {{5, 6, 7, 8}};
    Threefry4x32::key_type k = uk;
    auto a = u01all<float>(rng(c, k)); // returns std::array<float,4>
    for(auto e : a){
        std::cout << e << "\n";
    }
    c.incr();
    auto b = u01all<double>(rng(c, k));
    for(auto e : b){
        std::cout << e << "\n";
    }

    return 0;
}
#endif
