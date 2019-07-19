#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "spreadinterp.h"

using namespace std;
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "spreadinterp.h"

using namespace std;

int allocgpumemory1d(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	int ms = d_plan->ms;
	int nf1 = d_plan->nf1;
	int M = d_plan->M;

	d_plan->byte_now=0;
	// No extra memory is needed in idriven method;
	switch(opts.method)
	{
		case 5:
			{
				int numbins[1];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,  numbins[0]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,     numbins[0]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins[0]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,(numbins[0]+1)*sizeof(int)));
			}
			break;
	}
	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->c,M*sizeof(CUCPX)));

	checkCudaErrors(cudaMalloc((void**) &d_plan->fw, nf1*sizeof(CUCPX)));
	d_plan->fw_width = nf1;

	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fk,ms*sizeof(CUCPX)));

	return 0;
}

void freegpumemory1d(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaFree(d_plan->fw);
	cudaFree(d_plan->kx);
	cudaFree(d_plan->c);
	cudaFree(d_plan->fwkerhalf1);
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
	}
}

int allocgpumemory2d(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	int ms = d_plan->ms;
	int mt = d_plan->mt;
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int M = d_plan->M;
	int ntransfcufftplan = d_plan->ntransfcufftplan;

	d_plan->byte_now=0;
	// No extra memory is needed in idriven method;
	switch(opts.method)
	{
#if 0
		case 2:
			{
				//int total_mem_in_bytes=
				checkCudaErrors(cudaMalloc(&d_plan->kxsorted,M*sizeof(FLT)));
				checkCudaErrors(cudaMalloc(&d_plan->kysorted,M*sizeof(FLT)));
				checkCudaErrors(cudaMalloc(&d_plan->csorted,M*sizeof(CUCPX)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));

				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,(numbins[0]*
						numbins[1])*sizeof(int)));
			}
			break;
		case 4:
			{
				checkCudaErrors(cudaMalloc(&d_plan->kxsorted,M*sizeof(FLT)));
				checkCudaErrors(cudaMalloc(&d_plan->kysorted,M*sizeof(FLT)));
				checkCudaErrors(cudaMalloc(&d_plan->csorted,M*sizeof(CUCPX)));

				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,(numbins[0]*
						numbins[1]+1)*sizeof(int)));
			}
			break;
#endif
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

	//size_t pitch;
	checkCudaErrors(cudaMalloc(&d_plan->fw, ntransfcufftplan*nf1*nf2*
			sizeof(CUCPX)));

	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fk,ntransfcufftplan*ms*mt*
		sizeof(CUCPX)));

	d_plan->nstreams=16;
	cudaStream_t* streams =(cudaStream_t*) malloc(d_plan->nstreams*
		sizeof(cudaStream_t));
	for(int i=0; i<d_plan->nstreams; i++)
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
		case 2:
			{
				checkCudaErrors(cudaFree(d_plan->kxsorted));
				checkCudaErrors(cudaFree(d_plan->kysorted));
				checkCudaErrors(cudaFree(d_plan->csorted));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->temp_storage));
			}
			break;
		case 4:
			{
				checkCudaErrors(cudaFree(d_plan->kxsorted));
				checkCudaErrors(cudaFree(d_plan->kysorted));
				checkCudaErrors(cudaFree(d_plan->csorted));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->temp_storage));
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
	for(int i=0; i<d_plan->nstreams; i++)
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
		case 5:
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
			case 5:
			{
				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				numbins[1] = ceil((FLT) nf3/opts.bin_size_z);
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
	checkCudaErrors(cudaMalloc(&d_plan->fk,ntransfcufftplan*ms*mt*sizeof(CUCPX)));

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
	}
}
