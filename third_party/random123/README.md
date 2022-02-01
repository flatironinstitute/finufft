# Random123: a Library of Counter-Based Random Number Generators

<!-- Note that this file is both README.md and the doxygen mainpage.
     It should be minimally processed to uncomment the @ref directives
     before doxygen is run on it.  -->

The Random123 library is a collection of counter-based random
number generators (<!-- @ref CBRNG--> "CBRNGs") for CPUs (C and C++) and GPUs (CUDA and OpenCL), as described in
<a href="http://dl.acm.org/citation.cfm?doid=2063405"><i>Parallel Random Numbers:  As Easy
as 1, 2, 3</i>, Salmon, Moraes, Dror & Shaw, SC11, Seattle, Washington, USA, 2011, ACM </a>.
They are intended for use in statistical
applications and Monte Carlo simulation
and have passed all of the rigorous
SmallCrush, Crush and BigCrush tests in the 
<a href="http://www.iro.umontreal.ca/~simardr/testu01/tu01.html">
extensive TestU01 suite</a> of statistical tests for random number generators.
They are **not** suitable for use in cryptography or security
even though they are constructed using principles drawn from cryptography.

The Random123 library is implemented entirely in header files.
See [below](#installation-and-testing), for how to
install and use the library, and how to generate documentation
with doxygen.

CBRNGs are as fast as, or faster than conventional RNGs, much
easier to parallelize, use minimal memory/cache resources, and
require very little code.  On modern architectures, the
Random123 CBRNGs require a few cycles per byte of random data
returned and return random data in convenient sizes (arrays of
two or four elements, each of which is an unsigned integer of 32
or 64 bits).  The range of random numbers is the full
representable range of the 32 or 64 bit unsigned integer) 
The `<Random123/uniform.h>` header contains utility functions
to convert 32- and 64-bit unsigned integers to open or closed
ranges of single or double precision floating point numbers.

The Random123 library was written by John Salmon and Mark Moraes.
It is available at <a href="https://github.com/DEShawResearch/random123">
https://github.com/DEShawResearch/random123</a>.
Archived releases are also 
available from 
<a href="http://deshawresearch.com/resources_random123.html">
http://deshawresearch.com/resources_random123.html.</a>  Please see
the <!-- @ref LICENSE--> "LICENSE" for terms and conditions.

## Overview

Unlike conventional RNGs, counter-based RNGs are 
*stateless* functions (or function classes i.e. functors) whose
arguments are a *counter*, and a *key*
that return a result of the same type as the counter.

	result = CBRNGname(counter, key)

The returned result is a deterministic function of the key and counter,
i.e. a unique (counter, key) tuple will always produce the same
result.  The result is highly sensitive to small changes in the inputs,
so that the sequence of values produced by simply
incrementing the counter (or key) is effectively indistinguishable from a
sequence of samples of a uniformly distributed random variable.

For all the CBRNGs in the Random123 library, the result and
counter are the same type, specifically an array of *N* words,
where words have a width of *W* bits, encapsulated in 
<!-- @ref arrayNxW--> "r123arrayNxW" structs, or equivalently, for C++, in
the <!-- @ref r123::Array1x32--> "ArrayNxW" typedefs in the r123
namespace.   Keys are usually also arrayMxW types, but sometimes M is
a different size than the counter N (e.g. Philox keys have half the
number of elements as the counter, Threefry and ARS have the same number,
AES uses an opaque key type rather than an array)  The N random
numbers returned in `result.v[]` are unsigned integers of
width W (32 or 64), and the range of the random numbers is the full
range of the unsigned integer of that width (i.e. 0 to 2^W-1)

In C++, all public names (classes, structs, typedefs, etc) are in the
`r123` namespace.  In C, the public names (functions, enums, structs,
typedefs) begin either with `r123` or with one of the RNG family names, e.g.,
`threefry`, `philox`, `ars`, `aesni`.  The RNG functions themselves have names like
`philox4x32`.  C++ class names are capitalized, e.g., `Threefry4x32`.

## <!-- \anchor families--> The different families of Random123 generators

Several families of CBRNGs are available in this version of the library:
<ul>
<li> <!-- @ref ThreefryNxW--> "Threefry" is a **non-cryptographic**
adaptation of the Threefish block cipher from the <a href="http://www.skein-hash.info/"> Skein Hash Function</a>.  
See <!-- @ref--> r123::Threefry2x32, <!-- @ref--> r123::Threefry4x32, <!-- @ref--> r123::Threefry2x64, <!-- @ref--> r123::Threefry4x64.
<li> <!-- @ref PhiloxNxW--> "Philox" uses a Feistel network and integer multiplication.
See <!-- @ref--> r123::Philox2x32, <!-- @ref--> r123::Philox4x32, <!-- @ref--> r123::Philox2x64, <!-- @ref--> r123::Philox4x64.
The Nx64 forms are only available on hardware
that supports 64-bit multiplication producing a 128-bit result.
<li> <!@ref AESNI--> "AESNI" uses the Advanced Encryption Standard (AES) New Instruction,
available on certain modern x86 processors (some models of Intel Westmere and Sandy Bridge,
and AMD Interlagos, as of 2011).   AESNI CBRNGs can operate on four 32bit words (internally converting
them to the 128bit SSE type needed by the AES-NI instructions, or on a single m128i "word", 
which holds the SSE type.
See <!-- @ref--> r123::AESNI4x32, <!-- @ref--> r123::AESNI1xm128i.
<li> <!-- @ref AESNI--> "ARS" (Advanced Randomization System) is a **non-cryptographic** simplification of <!-- @ref AESNI--> "AESNI".
See <!-- @ref--> r123::ARS4x32_R, <!-- @ref--> r123::ARS1xm128i_R.
</ul>

## Installation and Testing

The Random123 library is implemented entirely in header files.  Thus,
there is nothing to compile before using it and nothing to link after
you `#include` it in your source files.  Simply direct your C or
C++ compiler to find the header files in the `include/` directory 
of the cloned repo and use the Random123
header files, types, and functions in your application.

Users and packagers are **STRONGLY ADVISED** run `make check` to
compile and run the tests in `tests/` before using Random123 in an
application (see <!-- @ref TestsREADME--> "tests/README").  Do not use
the library if any tests fail.  (It is not a failure for a test to
report that it cannot run because of missing hardware capabilities
like 64bit multiply, SSE, AES-NI or compiler capabilities)

The top-level GNUmakefile also has "install", "html", and
"install-html" targets.  The former will copy header files to
\$(DESTDIR)\$(includedir) (default: /usr/local/include).  The second
will run doxygen, replacing anything in docs/html.  The last will
install the documentation in \$(DESTDIR)\$(docdir)/html (default:
/usr/local/doc/Random123/html).

## Usage

### C++ API

A typical C++ use case might look like:


    #include <Random123/philox.h>

    typedef r123::Philox4x32 RNG;
    RNG rng;
    RNG::ctr_type c={{}};
    RNG::ukey_type uk={{}};
    uk[0] = ???; // some user_supplied_seed
    RNG::key_type k=uk;

    for(...){
       c[0] = ???; // some loop-dependent application variable 
       c[1] = ???; // another loop-dependent application variable 
       RNG::ctr_type r = rng(c, k);
       // use the random values in r for some operation related to
       // this iteration on objectid
    }

On each iteration, `r` contains an array of 4 32-bit random values that
will not be repeated by any other call to `rng` as long as `c` and `k`
are not reused.

In the example above, we use the <!-- @ref--> r123::Philox4x32, but any of the
other <!-- @ref CBRNG--> "CBRNGs" would serve equally well.  Also note that
for most CBRNGs, the `ukey_type` and the `key_type` are identical; the code
could just as well ignore the `ukey_type` and directly construct the
`key_type`.  However, for the <!-- @ref AESNI--> "AESNI" CBRNGs, the `key_type` is opaque, and
must be constructed from a `ukey_type`, as shown.

### The C API

In C, the example above could be written as:

    #include <Random123/philox.h>

    philox4x32_ctr_t c={{}};
    philox4x32_ukey_t uk={{}};

    uk.v[0] = user_supplied_seed;
    philox4x32_key_t k = philox4x32keyinit(uk);

    for(...){
        c.v[0] = ???; /* some loop-dependent application variable */
        c.v[1] = ???; /* another loop-dependent application variable */
        philox4x32_ctr_t r = philox4x32(c, k);
    }


In C, access to the contents of the counter and key is through
the fixed-size array member `v`.

## The CUDA platform

All relevant functions in the C and C++ APIs for Random123 are declared
as CUDA device functions if they are included in a CUDA kernel source file
and compiled with a CUDA compiler (nvcc).  They can be used exactly
as described/documented for regular C or C++ programs.  It is now
possible to use Random123 functions in 
both the host portion and the device portion of the same .cu source file.
The Nx32 forms were faster than the Nx64 variants on
32-bit GPU architectures in 2011, but we haven't measured this recently.

It has been reported that Random123 uses 16 bytes of
static memory per thread.  This is undesirable and not intentional,
but we do not have a workaround other than to suggest adjusting memory
allocation accordingly.

The
pi_cuda.cu and pi_cudapp.cu examples illustrate the use of CUDA.

In a machine with different GPUs, the
R123EXAMPLE_ENVCONF_CUDA_DEVICE environment variable can be set
to a unique substring of the CUDA GPU device name to select a
specific GPU (else examples try to choose the GPU with the most
cores)

## The OpenCL platform

The functions in the Random123 C API can all be used in
OpenCL kernels, just as in regular C functions. 
As with CUDA, the Nx32 forms are faster than the Nx64 variants on current (2011)
32-bit GPU architectures.

The `pi_opencl.c` and `pi_opencl_kernel.ocl` examples illustrate the use
of OpenCL.

In a machine with different OpenCL devices, the
R123EXAMPLE_ENVCONF_OPENCL_DEVICE environment variable can be
set to a unique substring of the OpenCL device name to select a
specific OpenCL device (else examples try to choose the device
with the most cores)

## C++11 \<random\> interface

In addition to the stateless ("pure/functional") C++ API above,
the Random123 package includes two C++ classes
that leverage the C++11 \<random\> API.

<ul>
<li>r123::MicroURNG provides an adapter class that provides a
more conventional interface compatible with the C++11 URNG
(uniform random number generator) API; the MicroURNG adapter can
be used with C++11 random number distributions and is
fast/lightweight enough that a new MicroURNG can be instantiated
with a unique key,counter tuple and used for each call to a
distribution, there is little or no overhead to creating
billions of unique MicroURNGs.  This adapter retains one of the
key advantages of CBRNGs -- complete application control over
the RNG state.
<li>r123::Engine provides the C++11 Random Engine API.  This can
also be used with any of the C++11 random distributions, but
sacrifices the application control over RNG state that is a
defining characteristic of CBRNGs.
</ul>

## The GNU Scientific Library (GSL) interface

In addition to the stateless ("pure/functional") C API above,
the Random123 package includes two C adapter interfaces
to the <a href="http://www.gnu.org/s/gsl/">GNU Scientific Library (GSL).</a>

<ul>
<li>The <!--\ref--> GSL_MICRORNG macro allows the application to
define a GSL random number generator.  It
can be used with GSL random distributions but still provides the
application with complete control over the RNG state (it is
analogous to the MicroURNG class, in that it uses shorter
periods, and is intended to be instantiated in large numbers for
a few calls to the random distribution).
<li>The <!--\ref--> GSL_CBRNG macro allows the application to create a GSL
RNG with a completely conventional interface, sacrificing
application control over the internal RNG state.
</ul>

##  Generating uniformly distributed and Gaussian distributed floats and doubles

The Random123 library provides generators for uniformly distributed
random **integers**.  Often, applications want random **real** values or
samples from other distributions.  The general problem of generating
samples from arbitrary distributions is beyond the scope of the Random123
library.  One can, of course, use GSL or MicroURNG and the
distributions in the C++11 \<random\> library, but a few simple cases
are common enough that all that extra machinery seems like overkill.
We have included a few generic conversion utilities which developers may
find useful.

<ul>
<li>    uniform.hpp - C++ functions that convert random integers to
         random, uniformly distributed floating point values.
<li>    u01fixedpt.h - C functions that convert random integers to
         random, uniformly distributed, equi-spaced, i.e., fixed point,
         values.
<li>    boxmuller.hpp - C++ functions that take two
         uniformly distributed integers (32 or 64 bit) and
         return a pair of Gaussian distributed floats or doubles.
</ul>

The Box-Muller method of generating Gaussian random variables is
particularly well suited to Random123 because it deterministically
consumes exactly two uniform randoms to generate exactly two gaussian
randoms.  It uses math library functions: sincos, log and sqrt which
may be slow on some platforms, but which are surprisingly fast on
others.  Notably, on GPUs, the lack of branching in the Box-Muller
method and hardware support for math functions overcomes the
transcendental function overhead, making it the fastest generator of
Gaussians that we are aware of.

## Examples

The <!-- @ref ExamplesREADME--> "examples/" directory, contains example code
intended to illustrate use of the library.

Complete, short programs estimate pi by counting the number of random
points that fall inside a circle inscribed in a square, demonstrating
the C, C++, AES, GSL, OpenCL, CUDA and C++11 APIs.  The environment
variable R123EXAMPLE_ENVCONF_SEED can be set to any unsigned integer value to run
the example with a different seed.  Many of the pi_* examples run different
numbers of iterations if that number is specified as the first argument
on the command line.

### Tests and Benchmarks

The <!-- @ref TestsREADME--> "tests/" directory contains tests and benchmarks.
This code is complicated due to the fact that it is
largely "single source" for CUDA, OpenCL and CPU implmentations.
Developers are strongly discouraged from emulating its style.
It contains:
<ul>
<li> Unit tests for individual components and "known-answer-tests", which
should be run to ensure that these RNGs build correctly on desired platforms.
These help to provide assurance that the code is being compiled correctly.
<li> A variety of timing harnesses are provided
which measure performance of a variety of generators in different
programming environments.
</ul>

## Portability

Although we have done our best to make Random123 portable and standards conforming,
it is an unfortunate fact that there is no portable code.  There is only
code that has been ported.

Prior to release, we test Random123 on a variety of systems and with a
variety of toolchains that are readily available to us.  Our
current test environment includes:

<ul>
<li> Linux, gcc-5.2.0, 6.3.0, 8.1.0, 10.1.0 using -march=native on Xeon hardware with
     AES and SSE4_2 and AVX2 support.
<li> Linux, gcc-5.2.0 using -m32.
<li> Linux, clang-8.0.0 with libc++ (8.0.0) on Xeon hardware.
<li> Linux, CentOS7 using the vendor supplied gcc toolchain (4.8.5-16).
<li> Linux, Ubuntu 16.04(LTS) using the vendor supplied gcc toolchain (5.4.0-6ubuntu1-16.04.11).
<li> Linux, Ubuntu 16.04(LTS) using OpenCL beignet 1.1.1-2 and https://github.com/intel/compute-runtime/releases/tag/19.07.12410
<li> Linux, Ubuntu 18.04(LTS) with clang-11.0.1 and both libc++ and libstdc++.
<li> Linux, icc-18.0.3 and 19.1.2.254 on Xeon hardware with AES, SSE4 and AVX2 support.
<li> Linux, NVIDIA CUDA 10.0.130 with GTX 980 and 1080, and Titan RTX (aka Turing) hardware.
<li> MacOS, with Xcode-10.1 and Metal on a 2018 Mac mini.
</ul>

In the past, we have tested Random123 with additional toolchains and
hardware.  Although we no longer test on these platforms, we know of
no reason that they should not work.

<ul>
<li>Linux, gcc (multiple versions from 3.4.3 through 6.3.0), on x86_64.
<li>Linux, gcc-4.1.2 and 4.4.1 on i686.
<li>Linux, gcc-4.8 on ARMv7 (32bit) Freescale/NXP LS1021A & ARM A53 (64bit) Freescale/NXP 1043A.
<li>Linux, clang-2.9, 3.0, 3.1, 3.3 and 3.6 on x86_64.
<li>Linux, clang-3.0 and 3.1 with lib++ (2012-04-19 svn checkout) on x86_64.
<li>Linux, clang-8.0.0 with libc++ on x86_64
<li>Linux, open64-4.2.4 on x86_64.
<li>Linux, Intel icc and icpc 12.0.2 on x86_64.
<li>Linux, NVIDIA CUDA 4.1.15, 4.2.6, 5.5.22 and 7.5.1.  (NOTE: We recommend against the use of CUDA before 4.1)
<li>Linux, OpenCL (NVIDIA SDK 4.0.17) on GTX480, M2090, GTX580 and GTX680 GPUs.
<li>Linux, OpenCL (AMD APP SDK 2.4 or 2.5), on x86_64 CPUs and Radeon HD6970 GPUs.
<li>Linux, OpenCL (Intel OpenCL 1.5), on x86_64 CPUs.
<li>Solaris, both gcc-3.4.3 and Sun C/C++ 5.8, on x86_64.
<li>FreeBSD 8.2, gcc-4.2.1, on x86_64.
<li>MacOS X 5.8, gcc-4.0.1, on i686.
<li>MacOS X 5.8, llvm-2.9.1 on i686 (problems with catching C++ exceptions).
<li>Windows 7, Microsoft Visual Studio, version 10.0, Microsoft C/C++ compiler 16.00.
</ul>

Others have reported success on
<ul>
<li>MacOS, OpenCL on x86_64 CPUs
<li>Linux, gcc-4.7.2 on Powerpc64 (BlueGene/Q)
<li>Linux, Portland Group Compiler on Powerpc64 (BlueGene/Q)
<li>Linux, IBM xlc on Powerpc64 (BlueGene/Q)
<li>MacOS, Metal on x86_64 CPUs and AMD Radeon R9 M380 GPU
<li>MacOS Sierra and Scientific Linux with Nvidia GPU and CUDA 8
<li>Linux, on s390x
</ul>

## Warnings

With some compilation options, the CUDA nvcc compiler warns about
unreachable code in array.h.  The compiler doesn't recognize that the
code that is unreachable for some values of some macro parameters, is
actually reachable for other values of the parameters.  It is possible
to disable that particular warning for a specific compilation unit by
adding -Wcudafe&nbsp;--diag_suppress=111 to the compilation command
line.

On our ARMv7 test platform, we suspect a compiler bug with -O3,
which does not seem to affect Random123 code itself, but
produces nondeterministic results from time_serial.  The
offending compiler version was
"aarch64-fsl-linux-gcc&nbsp;(Linaro&nbsp;GCC&nbsp;4.8-2014.04)&nbsp;4.8.3&nbsp;20140401&nbsp;(prerelease)".
We slightly reordered a couple of innocuous statements (a
timer() and dprintf() call) in time_serial to avoid the bug, but
we would avoid -O3 on ARMv7 with that particular version of the
compiler at least.

## Contributors

We welcome bug reports, bug fixes, ports,
general feedback and other enhancements to the
[issues](https://github.com/DEShawResearch/random123/issues) and 
[pull requests](https://github.com/DEShawResearch/random123/pulls) pages of
our [github repo](https://github.com/DEShawResearch/random123/).

We are grateful for contributed bug-fixes and portability enhancements from the following users:
<ul>
<li> Geoffrey Irving and Gabriel Rockefeller - BlueGene/Q and powerpc ports
<li> Yan Zhou - MacOS and clang ports
<li> David Lawrie - allowing 64-bit philox to compile for both host and device with CUDA
<li> Bogdan Opanchuk - pointing out the inconsistent rotation constants in the implementation of threefry2xW in version 1.07 and earlier.
<li> Tom Schoonjans - Support for Metal (Apple's successor to OpenCL)
<li> Karl Magdsick - documentation in uniform.hpp
<li> KT Thompson - Visual Studio 2015 and ibm xlc compiler ports
</ul>


