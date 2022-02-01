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
#ifndef demangle_dot_hpp_
#include <string>

// Every compiler has a demangling library *somewhere*.  Unfortunately, they're
// all different...

#ifdef __GNUC__
 // Clang defines __GNUC__, but clang3.1 with -stdlib=libc++ can't
 // find a <cxxabi.h> even though it *can* find the symbols at link
 // time.  I suspect this is a bug/oversight in the installation
 // process (which, in June 2012 is still pretty fluid for libc++), so
 // it might be fixed in the future.  On the other hand, the API in
 // cxxabi.h is locked down pretty tightly, so writing out an explicit
 // extern declaration is pretty safe, and avoids a rats nest of
 // ifdefs.  It is tempting to use clang's __has_include(<cxxabi.h>),
 // but it feels like more #ifdefs with no obvious upside.
 //
 // #include <cxxabi.h>
extern "C"{
  char*
  __cxa_demangle(const char* __mangled_name, char* __output_buffer,
		 size_t* __length, int* __status);
}
#endif
#include <typeinfo>

template <typename T>
std::string demangle(const T& ignored){
#ifdef __GNUC__
    int status;
    char *realname = __cxa_demangle(typeid(ignored).name(), 0, 0, &status);
    std::string ret;
    if(status!=0 || realname==0)
        ret = typeid(ignored).name();
    else
        ret = realname;
    free(realname);
    return ret;
#else
    return typeid(ignored).name();
#endif
}

#endif
