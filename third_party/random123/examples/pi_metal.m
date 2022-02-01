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

// functions to do boilerplate Metal begin and end
#include "../tests/util_metal.h"
#include "pi_check.h"

const char *progname;
int verbose = 0;
int debug = 0;

int
main(int argc, char **argv)
{
    unsigned count = argc > 1 ? atoi(argv[1]) : 0;
    UMetalInfo *infop;
    size_t i, nthreads = 1024, hits_sz;
    uint *hits_host;
    uint *tries_host;
    NSString *kernelname = @"counthits";
    NSError *err = nil;
    double d = 0.;
    id<MTLFunction> function;
    id<MTLBuffer> hits_dev, tries_dev;
    id<MTLCommandBuffer> buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pipeline;

    d = timer(&d);
    progname = argv[0];
    verbose = debug = argc > 2 ? atoi(argv[2]): 0;
    infop = metal_init(argc > 3 ? argv[3] : NULL, "pi_metal_kernel.metallib");
    CHECKNOTZERO(function = [infop->library newFunctionWithName: kernelname]);
    CHECKNOTZERO(hits_dev = [infop->device newBufferWithLength: sizeof(uint) * nthreads options:0]);
    CHECKNOTZERO(tries_dev = [infop->device newBufferWithLength: sizeof(uint) * nthreads options:0]);

    if (count == 0)
	count = NTRIES/nthreads;

    CHECKNOTZERO(buffer = [infop->queue commandBuffer]);
    CHECKNOTZERO(encoder = [buffer computeCommandEncoder]);
    CHECKERR(pipeline = [infop->device newComputePipelineStateWithFunction:function error:&err]);
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&count length:sizeof(uint) atIndex:0];
    [encoder setBuffer:hits_dev offset:0 atIndex:1];
    [encoder setBuffer:tries_dev offset:0 atIndex:2];

    MTLSize threadsPerThreadgroup = MTLSizeMake([pipeline maxTotalThreadsPerThreadgroup], 1, 1);
    MTLSize grid = MTLSizeMake(nthreads, 1, 1);

    [encoder dispatchThreads:grid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];

    hits_host = [hits_dev contents];
    tries_host = [tries_dev contents];

    unsigned long hits = 0, tries = 0;
    for (i = 0; i < nthreads; i++) {
	if (debug)
	    printf("%lu %u %u\n", (unsigned long)i, hits_host[i], tries_host[i]);
	hits += hits_host[i];
	tries += tries_host[i];
    }
    metal_done(infop);
    [function release];
    [hits_dev release];
    [tries_dev release];
    [buffer release];
    [encoder release];
    [pipeline release];
    return pi_check(hits, tries);
}
