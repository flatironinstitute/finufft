#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "spreadinterp.h"

using namespace std;

int allocgpumemory2d(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	int ms = d_plan->ms;
	int mt = d_plan->mt;
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int M = d_plan->M;
	int ntransfcufftplan = d_plan->ntransfcufftplan;

	d_plan->byte_now=0;
	// No extra memory is needed in idriven method (case 1);
	switch(opts.method)
	{
		case 5:
			{
				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
						(numbins[0]*numbins[1]+1)*sizeof(int)));
			}
		case 6:
			{
				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				checkCudaErrors(cudaMalloc(&d_plan->finegridsize,nf1*nf2*
						sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->fgstartpts,nf1*nf2*
						sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
						(numbins[0]*numbins[1]+1)*sizeof(int)));
			}
			break;
	}
	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->c,ntransfcufftplan*M*sizeof(CUCPX)));

	checkCudaErrors(cudaMalloc(&d_plan->fw, ntransfcufftplan*nf1*nf2*
			sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_plan->fk,ntransfcufftplan*ms*mt*
		sizeof(CUCPX)));

	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));

	cudaStream_t* streams =(cudaStream_t*) malloc(opts.nstreams*
		sizeof(cudaStream_t));
	for(int i=0; i<opts.nstreams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	d_plan->streams = streams;

	return 0;
}
void freegpumemory2d(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaFree(d_plan->fw);
	cudaFree(d_plan->kx);
	cudaFree(d_plan->ky);
	cudaFree(d_plan->c);
	cudaFree(d_plan->fwkerhalf1);
	cudaFree(d_plan->fwkerhalf2);
	switch(opts.method)
	{
		case 5:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->temp_storage));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
		case 6:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->finegridsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->temp_storage));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
	}

	for(int i=0; i<opts.nstreams; i++)
		checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));
}
