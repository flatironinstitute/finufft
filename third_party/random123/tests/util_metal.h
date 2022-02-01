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
#ifndef UTIL_METAL_H__
#define UTIL_METAL_H__
/*
 * has a couple of utility functions to setup and teardown Metal.
 * Avoid much boilerplate in every Metal program
 *
 * Written by Tom Schoonjans <Tom.Schoonjans@me.com>
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "util.h"

typedef struct umetal_info {
	id<MTLDevice> device;
	id<MTLCommandQueue> queue;
	id<MTLLibrary> library;
} UMetalInfo;

/* Miscellaneous checking macros for convenience */
static char *print_metal_errstring(NSError *err) {
    if (err == nil)
         return nil;
    return strdup(err.localizedDescription.UTF8String);
} 

#define CHECKERR(x) do { \
    (x); \
    if (err != nil) { \
	fprintf(stderr, "%s: error %s from %s\n", progname, print_metal_errstring(err), #x); \
	exit(1); \
    } \
} while(0)

//#define CHECK(x) CHECKERR(err = (x))

static UMetalInfo *metal_init(const char *devstr, const char *metallib)
{
    UMetalInfo *tp;
    NSError *err = nil;
    NSArray<id<MTLDevice>> *devices;
    unsigned i;
    unsigned long ndevices;

    /* get list of platforms */
    CHECKNOTZERO(devices = MTLCopyAllDevices());
    ndevices = (unsigned long) devices.count;
    dprintf(("ndevices = %lu\n", ndevices));
    if (ndevices == 0) {
	fprintf(stderr, "No Metal devices available\n");
	return NULL;
    }
    dprintf(("found %lu device%s\n", ndevices, ndevices == 1 ? "" : "s"));
    CHECKNOTZERO(tp = (UMetalInfo *) malloc(sizeof(UMetalInfo)));
    for (i = 0; i < devices.count; i++) {
	dprintf(("device %d: %s\n", i, devices[i].name.UTF8String));
    }

    // get default device
    tp->device = MTLCreateSystemDefaultDevice();

    dprintf(("create Metal command queue for device %s\n", tp->device.name.UTF8String));
    CHECKNOTZERO(tp->queue = [tp->device newCommandQueue]);
    dprintf(("create Metal library from %s\n", metallib));
    CHECKERR(tp->library = [tp->device newLibraryWithFile: [[NSString alloc] initWithUTF8String:metallib] error:&err]);

    return tp;
}


static void metal_done(UMetalInfo *tp) {
    
    dprintf(("metal_done\n"));
    [tp->library release];
    [tp->queue release];
    [tp->device release];
    free(tp);
}


#endif /* UTIL_METAL_H__ */
