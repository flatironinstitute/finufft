#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "spreadinterp.h"

using namespace std;

int allocgpumem2d_plan(cufinufft_plan *d_plan)
{
	int ms = d_plan->ms;
	int mt = d_plan->mt;
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int ntransfcufftplan = d_plan->ntransfcufftplan;

	d_plan->byte_now=0;
	// No extra memory is needed in idriven method (case 1)
	switch(d_plan->opts.gpu_method)
	{
		case 5:
			{
				int numbins[2];
				numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
				numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
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
				numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
				numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
				checkCudaErrors(cudaMalloc(&d_plan->finegridsize,nf1*nf2*
						sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->fgstartpts,nf1*nf2*
						sizeof(int)));
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

	checkCudaErrors(cudaMalloc(&d_plan->fw, ntransfcufftplan*nf1*nf2*
			sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_plan->fk,ntransfcufftplan*ms*mt*
		sizeof(CUCPX)));

	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));

	cudaStream_t* streams =(cudaStream_t*) malloc(d_plan->opts.gpu_nstreams*
		sizeof(cudaStream_t));
	for(int i=0; i<d_plan->opts.gpu_nstreams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	d_plan->streams = streams;

	return 0;
}

int allocgpumem2d_nupts(cufinufft_plan *d_plan)
{
	int M = d_plan->M;
	int ntransfcufftplan = d_plan->ntransfcufftplan;

	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->c,ntransfcufftplan*M*sizeof(CUCPX)));
	switch(d_plan->opts.gpu_method)
	{
		case 5:
		case 6:
			{
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
			}
			break;
	}
	return 0;
}

void freegpumemory2d(cufinufft_plan *d_plan)
{
	cudaFree(d_plan->fw);
	cudaFree(d_plan->kx);
	cudaFree(d_plan->ky);
	cudaFree(d_plan->c);
	cudaFree(d_plan->fwkerhalf1);
	cudaFree(d_plan->fwkerhalf2);
	switch(d_plan->opts.gpu_method)
	{
		case 5:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
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
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
	}

	for(int i=0; i<d_plan->opts.gpu_nstreams; i++)
		checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));
}
int allocgpumem1d_plan(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
int allocgpumem1d_nupts(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
void freegpumemory1d(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
}

int allocgpumem3d_plan(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
int allocgpumem3d_nupts(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
void freegpumemory3d(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
}
