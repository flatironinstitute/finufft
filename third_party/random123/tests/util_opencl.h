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
#ifndef UTIL_OPENCL_H__
#define UTIL_OPENCL_H__
/*
 * has a couple of utility functions to setup and teardown OpenCL.
 * Avoid much boilerplate in every OpenCL program
 */

#include "util.h"

/* We use the old clCreateCommandQueue for max portability since we do not
 * use any cmdq properties anyway, avoids a warning.
 */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#define UCL_STRSIZE 128

typedef struct ucl_info {
    cl_context ctx;
    cl_program prog;
    cl_device_id devid;
    cl_command_queue cmdq;
    cl_uint clkfreq, compunits;
    size_t wgsize;
    int cores;
    double cycles;
    char vendor[UCL_STRSIZE], devname[UCL_STRSIZE],
	version[UCL_STRSIZE], driver[UCL_STRSIZE];
    int computeflags;
    cl_device_fp_config fpdbl;
    cl_device_type devtype;
} UCLInfo;

/* Miscellaneous checking macros for convenience */
static char *print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
} 

static const char *cldevtypestr(cl_device_type c) {
    switch (c) {
    case CL_DEVICE_TYPE_CPU: return "CPU";
    case CL_DEVICE_TYPE_GPU: return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR: return "ACCELERATOR";
    case CL_DEVICE_TYPE_DEFAULT: return "DEFAULT";
    default: return "UNKNOWN";
    }
}

#define CHECKERR(x) do { \
    (x); \
    if (err != CL_SUCCESS) { \
	fprintf(stderr, "%s: error %d: %s from %s\n", progname, err, print_cl_errstring(err), #x); \
	exit(1); \
    } \
} while(0)

#define CHECK(x) CHECKERR(err = (x))

static UCLInfo *opencl_init(const char *devstr, const char *src,
			    const char *options)
{
#define UCL_MAX_PROPERTIES 32
#define UCL_MAX_PLATFORMS 8
#define UCL_MAX_DEVICES 16
    UCLInfo *tp;
    cl_context_properties ctxprop[UCL_MAX_PROPERTIES];
    cl_int err;
    cl_platform_id platforms[UCL_MAX_PLATFORMS];
    cl_uint nplatforms, ndevices;
    cl_device_id devices[UCL_MAX_DEVICES];
    const char *srcstr[2], *clbinfile, *coremultstr;
    unsigned i, j;
    int cores, devcores, coremultguess = 0;

    if (devstr == NULL)
	devstr = getenv("R123EXAMPLE_ENVCONF_OPENCL_DEVICE");

    coremultstr = getenv("R123EXAMPLE_ENVCONF_OPENCL_CORES_PER_UNIT");
    if (coremultstr) {
	coremultguess = atoi(coremultstr);
	dprintf(("setting coremultguess to %d\n", coremultguess));
    }
	
    /* get list of platforms */
    CHECK(clGetPlatformIDs(0, NULL, &nplatforms));
    dprintf(("nplatforms = %d\n", nplatforms));
    CHECK(clGetPlatformIDs(UCL_MAX_PLATFORMS, platforms, &nplatforms));
    if (nplatforms == 0) {
	fprintf(stderr, "No OpenCL platforms available\n");
	return NULL;
    }
    dprintf(("found %d platform%s:\n", nplatforms, nplatforms == 1 ? "" : "s"));
    CHECKNOTZERO(tp = (UCLInfo *) malloc(sizeof(UCLInfo)));
    ctxprop[0] = CL_CONTEXT_PLATFORM;
    ctxprop[1] = 0; /* will fill in platform in loop */
    ctxprop[2] = 0;
    cores = devcores = 0;
    for (i = 0; i < nplatforms; i++) {
	dprintf(("platform %d: 0x%lx\n", i, (unsigned long)platforms[i]));
	clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, UCL_MAX_DEVICES,
		       devices, &ndevices);
	// Sometimes, a platform with no devices will just return a failure instead
	// of 0 devices, do not be deterred by that ...
	if (err != CL_SUCCESS) {
            fprintf(stderr, "%s: error %d: %s from clGetDeviceIDs for platform %d ( 0x%lx )\n", progname, err, print_cl_errstring(err), i,
		    (unsigned long)platforms[i]);
	    continue;
	}
	dprintf(("platform 0x%lx has %d devices:\n", (unsigned long)platforms[i], ndevices));
	for (j = 0; j < ndevices; j++) {
	    UCLInfo uc;
	    uc.devid = devices[j];
	    CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_NAME,
				  sizeof uc.devname, uc.devname, 0));
	    CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR,
				  sizeof uc.vendor, uc.vendor, 0));
	    CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_VERSION,
				  sizeof uc.version, uc.version, 0));
	    CHECK(clGetDeviceInfo(devices[j], CL_DRIVER_VERSION,
				  sizeof uc.driver, uc.driver, 0));
	    CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
				  sizeof uc.clkfreq, &uc.clkfreq, 0));
	    CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
				  sizeof uc.compunits, &uc.compunits, 0));
	    CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
				  sizeof uc.wgsize, &uc.wgsize, 0));
	    CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_DOUBLE_FP_CONFIG,
				  sizeof uc.fpdbl, &uc.fpdbl, 0));
	    CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_TYPE,
				  sizeof uc.devtype, &uc.devtype, 0));
	    uc.computeflags = 0;
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
	    {
		cl_uint nvmaj, nvmin;
		if(clGetDeviceInfo(devices[j], CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,
				   sizeof(nvmaj), &nvmaj, 0) == CL_SUCCESS &&
		   clGetDeviceInfo(devices[j], CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,
				   sizeof(nvmin), &nvmin, 0) == CL_SUCCESS) {
		    uc.computeflags = nvmaj*10 + nvmin;
		}
	    }
#endif
	    cores = uc.compunits;
	    /* XXX Hardwired knowledge about devices */
	    if (coremultguess) {
		cores *= coremultguess;
		printf("Using override cores per unit %d so %d cores\n", coremultguess, cores);
	    } else if (strstr(uc.devname, "Cayman") || strstr(uc.devname, "Tahiti")) {
		/*
		 * Most modern AMD compute units (shader cluster?)
		 * are a 16-lane SIMD Engine with 4 (Cayman) or 5
		 * (Cypress) VLIW slots per lane.  AMD appears to
		 * think of each slot as a "stream processor"
		 * (shader processor) in their marketing i.e. a
		 * Cayman-based Radeon 6950 with 24 compute units
		 * has 1536 stream processors.
		 * With Tahiti/Southern Islands/GCN, each compute
		 * unit has four vector execution SIMD units, each
		 * with 16 lanes.  So the Tahiti-based Radeon 7970 with
		 * 32 compute units has 2048 cores/stream processors.
		 */
		cores *= 16*4;
	    } else if (strstr(uc.devname, "Cypress")) {
		cores *= 16*5;
	    } else if (strstr(uc.devname, "GTX TITAN X") ||
		       strstr(uc.devname, "GTX 9")) {
		cores *= 128;
	    } else if (strstr(uc.devname, "GTX 6") ||
		       strstr(uc.devname, "GTX 7") ||
		       strstr(uc.devname, "Tesla K2") ||
		       strstr(uc.devname, "Tesla K4") ||
		       strstr(uc.devname, "GTX TITAN")) {
		/* Kepler has 192 cores per SMX */
		cores *= 192;
	    } else if (strstr(uc.devname, "GTX 580") ||
		       strstr(uc.devname, "GTX 480") ||
		       strstr(uc.devname, "C20") ||
		       strstr(uc.devname, "M20")) {
		/*
		 * Fermi has 32 cores per SM.  Maybe use
		 * computeflags to figure this out?
		 */
		cores *= 32;
	    } else if (uc.devtype == CL_DEVICE_TYPE_GPU) {
		fprintf(stderr, "Unknown # of cores per unit for this device \"%s\", assuming 1, so cpb may be wrong and choice of threads may be suboptimal, fix by setting R123EXAMPLE_ENVCONF_OPENCL_CORES_PER_UNIT\n",
			uc.devname);
	    }
	    /* clkfreq is in Megahertz! */
	    uc.cycles = 1e6 * uc.clkfreq * cores;
	    dprintf(("  %d: device 0x%lx vendor %s %s version %s driver %s : %u compute units @ %u MHz %d cores cycles/s %.2f flags %d fpdbl 0x%lx\n",
		     j, (unsigned long) devices[j], uc.vendor, uc.devname, uc.version,
		     uc.driver, uc.compunits, uc.clkfreq, cores, uc.cycles, uc.computeflags,
		     (unsigned long) uc.fpdbl));
	    if (devstr && strstr(uc.devname, devstr) == NULL) {
		if (verbose || debug)
		    printf("skipping device %s\n", uc.devname);
		continue;
	    }
	    if (cores > devcores) {
		ctxprop[1] = (cl_context_properties) platforms[i];
		devcores = cores;
		*tp = uc;
	    }
	}
    }
    if (devcores == 0) {
	fprintf(stderr, "%s: No matching devices found\n", progname);
	exit(1);
    }
    tp->cores = devcores;

    // using DEVICE_MAX_WORKGROUP_SIZE as the workgroup size seems to break
    // weirdly on NVIDIA SDK 4.0.17 (the returned ctr arrays are all zeros)
    // Halving it seems to produce as good or fractionally better performance
    // on AMD, so seems a good choice. -- mm, 20110831
    if (tp->wgsize > 2) {
	tp->wgsize /= 2;
    }
    printf("device 0x%lx %s : %d units %d cores %.2f Gcycles/s %lu maxwg %s device\n",
	     (unsigned long)tp->devid, tp->devname, tp->compunits, devcores, tp->cycles*1e-9, tp->wgsize, cldevtypestr(tp->devtype));
    CHECKERR(tp->ctx = clCreateContext(ctxprop, 1, &tp->devid, 0, 0, &err));
    dprintf(("create OpenCL context for device 0x%lx %s\n", (unsigned long)tp->devid, tp->devname));
    CHECKERR(tp->cmdq = clCreateCommandQueue(tp->ctx, tp->devid, 0, &err));
    /*
     * create & compile OpenCL program from source string.  Could
     * normalize this out of the context but that creates a more
     * complex API.
     */
    dprintf(("create OpenCL program from source\n"));

    /* If the device has support for double, enable it, might need it for u01.h */
    i = 0;
#define UCLDBL "\n\
#ifdef cl_khr_fp64\n\
#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\
#elif defined(cl_amd_fp64)\n\
#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n\
#endif\n\
"
    if (tp->fpdbl) {
	srcstr[i++] = UCLDBL;
    }
    srcstr[i++] = src;
    CHECKERR(tp->prog = clCreateProgramWithSource(tp->ctx, i, srcstr, 0, &err));
    if ((err = clBuildProgram(tp->prog, 1, &tp->devid, options, 0, 0)) != CL_SUCCESS || debug) {
	char errbuf[512*1024];
	strcpy(errbuf, "<error report not filled in>");
	cl_int builderr = err;
	CHECK(clGetProgramBuildInfo(tp->prog, tp->devid, CL_PROGRAM_BUILD_LOG,
				    sizeof errbuf, &errbuf[0], 0));
	fprintf(stderr, "%s: OpenCL build for device id 0x%lx %s returned error %d (%s): %s\n",
		progname, (unsigned long) tp->devid, tp->devname, builderr, print_cl_errstring(builderr), errbuf);
	if (builderr != CL_SUCCESS)
	    exit(1);
    }
    if ((clbinfile = getenv("R123_SAVE_OPENCL_BINARY")) != NULL) {
	size_t sz, szret;
	unsigned char *binp;
	FILE *fp;
	CHECKERR(clGetProgramInfo(tp->prog, CL_PROGRAM_BINARY_SIZES, sizeof(sz), &sz, &szret));
	CHECKNOTZERO(szret);
	CHECKNOTZERO(sz);
	printf("szret %lu, sz %lu\n", szret, (unsigned long) sz);
	if (szret > 0 && sz > 0) {
	    CHECKNOTZERO((binp = (unsigned char *) malloc(sz)));
	    CHECKERR(clGetProgramInfo(tp->prog, CL_PROGRAM_BINARIES, sizeof(binp), &binp, &szret));
	    CHECKNOTZERO(szret);
	    CHECKNOTZERO(fp = fopen(clbinfile, "wc"));
	    CHECKEQUAL(sz, fwrite(binp, 1, sz, fp));
	    CHECKZERO(fclose(fp));
	    free(binp);
	    printf("wrote OpenCL binary to %s\n", clbinfile);
	}
    }
    dprintf(("opencl_init done\n"));
    /* XXX Save build programs as .deviceid so we can read them back and run? */
    return tp;
}


static void opencl_done(UCLInfo *tp) {
    cl_int err;
    
    dprintf(("opencl_done\n"));
    CHECK(clReleaseCommandQueue(tp->cmdq));
    tp->cmdq = 0;
    CHECK(clReleaseProgram(tp->prog));
    tp->prog = 0;
    CHECK(clReleaseContext(tp->ctx));
    tp->ctx = 0;
    free(tp);
}


#endif /* UTIL_OPENCL_H__ */
