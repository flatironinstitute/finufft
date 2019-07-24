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

int allocgpumemory3d(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	int ms = d_plan->ms;
	int mt = d_plan->mt;
	int mu = d_plan->mu;
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int M = d_plan->M;
	int ntransfcufftplan = d_plan->ntransfcufftplan;

	d_plan->byte_now=0;
	// No extra memory is needed in idriven method;
	switch(opts.method)
	{
		case 6:
			{
				int numobins[3], numbins[3];
				numobins[0] = ceil((FLT) nf1/opts.o_bin_size_x);
				numobins[1] = ceil((FLT) nf2/opts.o_bin_size_y);
				numobins[2] = ceil((FLT) nf3/opts.o_bin_size_z);

				numbins[0] = numobins[0]*opts.o_bin_size_x/opts.bin_size_x;
				numbins[1] = numobins[1]*opts.o_bin_size_y/opts.bin_size_y;
				numbins[2] = numobins[2]*opts.o_bin_size_z/opts.bin_size_z;
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,
					numobins[0]*numobins[1]*numobins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numnupts,
					numobins[0]*numobins[1]*numobins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,
					numbins[0]*numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,
					numbins[0]*numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,(numobins[0]
					*numobins[1]*numobins[2]+1)*sizeof(int)));
			}
			break;
		case 1:
		case 2:
		case 3:
			{
				int numobins[3], numbins[3];
				int binsperobins[3];
				numobins[0] = ceil((FLT) nf1/opts.o_bin_size_x);
				numobins[1] = ceil((FLT) nf2/opts.o_bin_size_y);
				numobins[2] = ceil((FLT) nf3/opts.o_bin_size_z);

				binsperobins[0] = opts.o_bin_size_x/opts.bin_size_x;
				binsperobins[1] = opts.o_bin_size_y/opts.bin_size_y;
				binsperobins[2] = opts.o_bin_size_z/opts.bin_size_z;

				numbins[0] = numobins[0]*(binsperobins[0]+2);
				numbins[1] = numobins[1]*(binsperobins[1]+2);
				numbins[2] = numobins[2]*(binsperobins[2]+2);

				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,
					numobins[0]*numobins[1]*numobins[2]*sizeof(int)));
				//checkCudaErrors(cudaMalloc(&d_plan->numnupts,
					//numobins[0]*numobins[1]*numobins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,
					numbins[0]*numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,
					(numbins[0]*numbins[1]*numbins[2]+1)*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,(numobins[0]
					*numobins[1]*numobins[2]+1)*sizeof(int)));
			}
			break;
		case 4:
			{
				if(opts.sort){
					int numbins[3];
					numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
					numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
					numbins[2] = ceil((FLT) nf3/opts.bin_size_z);
					checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
						numbins[1]*numbins[2]*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*numbins[2]*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*numbins[2]*sizeof(int)));
				}else{
					checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				}
			}
			break;
		case 5:
			{
				int numbins[3];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				numbins[2] = ceil((FLT) nf3/opts.bin_size_z);
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
					numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
					numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
					numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
					(numbins[0]*numbins[1]*numbins[2]+1)*sizeof(int)));
			}
			break;
	}
	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->kz,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->c,ntransfcufftplan*M*sizeof(CUCPX)));

	//size_t pitch;
	checkCudaErrors(cudaMalloc(&d_plan->fw, ntransfcufftplan*nf1*nf2*nf3*
		sizeof(CUCPX)));

	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf3,(nf3/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fk,ntransfcufftplan*ms*mt*mu*
		sizeof(CUCPX)));

	d_plan->nstreams=min(16, d_plan->ntransfcufftplan);
	cudaStream_t* streams =(cudaStream_t*) malloc(d_plan->nstreams*
		sizeof(cudaStream_t));
	for(int i=0; i<d_plan->nstreams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	d_plan->streams = streams;

	return 0;
}
void freegpumemory3d(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaFree(d_plan->fw);
	cudaFree(d_plan->kx);
	cudaFree(d_plan->ky);
	cudaFree(d_plan->kz);
	cudaFree(d_plan->c);
	cudaFree(d_plan->fwkerhalf1);
	cudaFree(d_plan->fwkerhalf2);
	cudaFree(d_plan->fwkerhalf3);
	switch(opts.method)
	{
		case 6:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
				checkCudaErrors(cudaFree(d_plan->subprob_to_nupts));
			}
			break;
		case 1:
		case 2:
		case 3:
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
		case 4:
			{
				if(opts.sort){
					checkCudaErrors(cudaFree(d_plan->idxnupts));
					checkCudaErrors(cudaFree(d_plan->sortidx));
					checkCudaErrors(cudaFree(d_plan->binsize));
					checkCudaErrors(cudaFree(d_plan->binstartpts));
				}else{
					checkCudaErrors(cudaFree(d_plan->idxnupts));
				}
			}
			break;
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
	}
	for(int i=0; i<d_plan->nstreams; i++)
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
