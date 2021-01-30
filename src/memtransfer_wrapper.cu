#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "memtransfer.h"

using namespace std;

int ALLOCGPUMEM2D_PLAN(CUFINUFFT_PLAN d_plan)
/*
	wrapper for gpu memory allocation in "plan" stage.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int maxbatchsize = d_plan->maxbatchsize;

	d_plan->byte_now=0;
	// No extra memory is needed in nuptsdriven method (case 1)
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort){
					int numbins[2];
					numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
					numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
					checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*sizeof(int)));
				}
			}
			break;
		case 2:
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
			break;
		case 3:
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
		default:
			cerr << "err: invalid method " << endl;
	}

	if(!d_plan->opts.gpu_spreadinterponly){
		checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize*nf1*nf2*
				sizeof(CUCPX)));
		checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
		checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));
	}

	cudaStream_t* streams =(cudaStream_t*) malloc(d_plan->opts.gpu_nstreams*
		sizeof(cudaStream_t));
	for(int i=0; i<d_plan->opts.gpu_nstreams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	d_plan->streams = streams;

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
	return 0;
}

int ALLOCGPUMEM2D_NUPTS(CUFINUFFT_PLAN d_plan)
/*
	wrapper for gpu memory allocation in "setNUpts" stage.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	int M = d_plan->M;

	if(d_plan->sortidx ) checkCudaErrors(cudaFree(d_plan->sortidx));
	if(d_plan->idxnupts) checkCudaErrors(cudaFree(d_plan->idxnupts));

	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort)
					checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
			}
			break;
		case 2:
		case 3:
			{
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
			}
			break;
		default:
			cerr<<"err: invalid method" << endl;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}

void FREEGPUMEMORY2D(CUFINUFFT_PLAN d_plan)
/*
	wrapper for freeing gpu memory.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	if(!d_plan->opts.gpu_spreadinterponly){
		checkCudaErrors(cudaFree(d_plan->fw));
		checkCudaErrors(cudaFree(d_plan->fwkerhalf1));
		checkCudaErrors(cudaFree(d_plan->fwkerhalf2));
	}
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort){
					checkCudaErrors(cudaFree(d_plan->idxnupts));
					checkCudaErrors(cudaFree(d_plan->sortidx));
					checkCudaErrors(cudaFree(d_plan->binsize));
					checkCudaErrors(cudaFree(d_plan->binstartpts));
				}else{
					checkCudaErrors(cudaFree(d_plan->idxnupts));
				}
			}
			break;
		case 2:
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
		case 3:
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

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
}

int ALLOCGPUMEM1D_PLAN(CUFINUFFT_PLAN d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
int ALLOCGPUMEM1D_NUPTS(CUFINUFFT_PLAN d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
void FREEGPUMEMORY1D(CUFINUFFT_PLAN d_plan)
{
	cerr<<"Not yet implemented"<<endl;
}

int ALLOCGPUMEM3D_PLAN(CUFINUFFT_PLAN d_plan)
/*
	wrapper for gpu memory allocation in "plan" stage.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int maxbatchsize = d_plan->maxbatchsize;

	d_plan->byte_now=0;

	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort){
					int numbins[3];
					numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
					numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
					numbins[2] = ceil((FLT) nf3/d_plan->opts.gpu_binsizez);
					checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*numbins[2]*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*numbins[2]*sizeof(int)));
				}
			}
			break;
		case 2:
			{
				int numbins[3];
				numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
				numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
				numbins[2] = ceil((FLT) nf3/d_plan->opts.gpu_binsizez);
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
		case 4:
			{
				int numobins[3], numbins[3];
				int binsperobins[3];
				numobins[0] = ceil((FLT) nf1/d_plan->opts.gpu_obinsizex);
				numobins[1] = ceil((FLT) nf2/d_plan->opts.gpu_obinsizey);
				numobins[2] = ceil((FLT) nf3/d_plan->opts.gpu_obinsizez);

				binsperobins[0] = d_plan->opts.gpu_obinsizex/
					d_plan->opts.gpu_binsizex;
				binsperobins[1] = d_plan->opts.gpu_obinsizey/
					d_plan->opts.gpu_binsizey;
				binsperobins[2] = d_plan->opts.gpu_obinsizez/
					d_plan->opts.gpu_binsizez;

				numbins[0] = numobins[0]*(binsperobins[0]+2);
				numbins[1] = numobins[1]*(binsperobins[1]+2);
				numbins[2] = numobins[2]*(binsperobins[2]+2);

				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,
					numobins[0]*numobins[1]*numobins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,
					numbins[0]*numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,
					(numbins[0]*numbins[1]*numbins[2]+1)*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,(numobins[0]
					*numobins[1]*numobins[2]+1)*sizeof(int)));
			}
			break;
		default:
			cerr << "err: invalid method" << endl;
	}

	if(!d_plan->opts.gpu_spreadinterponly){
		checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize*nf1*nf2*nf3*
			sizeof(CUCPX)));
		checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
		checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));
		checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf3,(nf3/2+1)*sizeof(FLT)));
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}

int ALLOCGPUMEM3D_NUPTS(CUFINUFFT_PLAN d_plan)
/*
	wrapper for gpu memory allocation in "setNUpts" stage.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	int M = d_plan->M;

	d_plan->byte_now=0;

	if(d_plan->sortidx ) checkCudaErrors(cudaFree(d_plan->sortidx));
	if(d_plan->idxnupts) checkCudaErrors(cudaFree(d_plan->idxnupts));

	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort)
					checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
			}
			break;
		case 2:
			{
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
			}
			break;
		case 4:
			{
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
			}
			break;
		default:
			cerr << "err: invalid method" << endl;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}
void FREEGPUMEMORY3D(CUFINUFFT_PLAN d_plan)
/*
	wrapper for freeing gpu memory.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);


	if(!d_plan->opts.gpu_spreadinterponly){
		cudaFree(d_plan->fw);
		cudaFree(d_plan->fwkerhalf1);
		cudaFree(d_plan->fwkerhalf2);
		cudaFree(d_plan->fwkerhalf3);
	}

	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort){
					checkCudaErrors(cudaFree(d_plan->idxnupts));
					checkCudaErrors(cudaFree(d_plan->sortidx));
					checkCudaErrors(cudaFree(d_plan->binsize));
					checkCudaErrors(cudaFree(d_plan->binstartpts));
				}else{
					checkCudaErrors(cudaFree(d_plan->idxnupts));
				}
			}
			break;
		case 2:
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
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
	}

	for(int i=0; i<d_plan->opts.gpu_nstreams; i++)
		checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
}
