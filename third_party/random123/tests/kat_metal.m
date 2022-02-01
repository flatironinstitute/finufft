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
// Simple Metal device kernel and host main program to
// compute pi via random darts at a square
//
// Written by Tom Schoonjans <Tom.Schoonjans@me.com>

// We're compiling on the host, so we don't include metalfeatures.h,
// but metal doesn't have 64-bit arithmetic, so we turn it off here.
#define R123_USE_64BIT 0
// functions to do boilerplate Metal begin and end
#include "../tests/util_metal.h"
#include "kat_main.h"

void host_execute_tests(kat_instance *tests, unsigned ntests){
    UMetalInfo *infop;
    size_t i, nthreads = 1024, hits_sz;
    uint *tests_host;
    NSString *kernelname = @"dev_execute_tests";
    NSError *err = nil;
    id<MTLFunction> function;
    id<MTLBuffer> tests_dev;
    size_t tests_sz;
    id<MTLCommandBuffer> buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pipeline;

    infop = metal_init(NULL, "kat_metal_kernel.metallib");
    CHECKNOTZERO(function = [infop->library newFunctionWithName: kernelname]);
    tests_sz = sizeof(kat_instance) * (ntests+1); // +1 for sentinel test with method==last
    CHECKNOTZERO(tests_dev = [infop->device newBufferWithBytes: tests length: tests_sz options:0]);

    CHECKNOTZERO(buffer = [infop->queue commandBuffer]);
    CHECKNOTZERO(encoder = [buffer computeCommandEncoder]);
    CHECKERR(pipeline = [infop->device newComputePipelineStateWithFunction:function error:&err]);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:tests_dev offset:0 atIndex:0];

    MTLSize threadsPerThreadgroup = MTLSizeMake([pipeline maxTotalThreadsPerThreadgroup], 1, 1);
    MTLSize grid = MTLSizeMake(nthreads, 1, 1);

    [encoder dispatchThreads:grid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];

    tests_host = [tests_dev contents];

    memcpy(tests, tests_host, tests_sz);
    metal_done(infop);
    [function release];
    [tests_dev release];
    [buffer release];
    [encoder release];
    [pipeline release];
}
