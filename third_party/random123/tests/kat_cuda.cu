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
#include "util_cuda.h"
#include "kat_main.h"

#define KAT_KERNEL __global__
#define KAT_GLOBAL
#include "kat_dev_execute.h"

void host_execute_tests(kat_instance *tests_host, unsigned ntests){
    CUDAInfo *infop;
    kat_instance *tests_dev;
    size_t tests_sz;

    infop = cuda_init(NULL);

    tests_sz = sizeof(tests_host[0]) * (ntests+1); // +1 for sentinel test with method==last
    CHECKCALL(cudaMalloc(&tests_dev, tests_sz));
    CHECKCALL(cudaMemcpy(tests_dev, tests_host, tests_sz, cudaMemcpyHostToDevice));

    printf("starting %u tests on 1 blocks with 1 threads/block\n", ntests);
    fflush(stdout);

    // TO DO:  call this with parallelism, <<<infop->blocks_per_grid, infop->threads_per_block>>>
    // and then insure that each of the threads got the same result.
    dev_execute_tests<<<1, 1>>>(tests_dev);

    CHECKCALL(cudaDeviceSynchronize());
    CHECKCALL(cudaMemcpy(tests_host, tests_dev, tests_sz, cudaMemcpyDeviceToHost));
    CHECKCALL(cudaFree(tests_dev));
    cuda_done(infop);
}

