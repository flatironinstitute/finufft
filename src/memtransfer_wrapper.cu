#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "spread.h"

using namespace std;

int allocgpumemory(spread_opts opts, cufinufft_plan *d_plan)
{
	int ms = d_plan->ms;
	int mt = d_plan->mt;
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int M = d_plan->M;

	d_plan->byte_now=0;
	// No extra memory is needed in idriven method;
	switch(opts.method)
	{
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
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,(numbins[0]*numbins[1])*sizeof(int)));
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
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,(numbins[0]*numbins[1]+1)*sizeof(int)));
			}
			break;
		case 5:
			{
				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,  numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,     numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts, numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,(numbins[0]*numbins[1]+1)*sizeof(int)));
			}
			break;
	}
	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->c,M*sizeof(CUCPX)));

	size_t pitch;
	checkCudaErrors(cudaMallocPitch((void**) &d_plan->fw, &pitch,nf1*sizeof(CUCPX),nf2));
	d_plan->fw_width = pitch/sizeof(CUCPX);

	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fk,ms*mt*sizeof(CUCPX)));

	return 0;
}

int copycpumem_to_gpumem(cufinufft_plan *d_plan)
{
	int M=d_plan->M;
	int nf1=d_plan->nf1;
	int nf2=d_plan->nf2;
	checkCudaErrors(cudaMemcpy(d_plan->kx,d_plan->h_kx,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->ky,d_plan->h_ky,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->c, d_plan->h_c,M*sizeof(CUCPX),cudaMemcpyHostToDevice));

	if(d_plan->h_fwkerhalf1 != NULL)
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf1,d_plan->h_fwkerhalf1,(nf1/2+1)*sizeof(FLT),cudaMemcpyHostToDevice));
	if(d_plan->h_fwkerhalf2 != NULL)
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf2,d_plan->h_fwkerhalf2,(nf2/2+1)*sizeof(FLT),cudaMemcpyHostToDevice));
	
	return 0;
}

int copygpumem_to_cpumem_fw(cufinufft_plan *d_plan)
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int fw_width = d_plan->fw_width;
	checkCudaErrors(cudaMemcpy2D(d_plan->h_fw,nf1*sizeof(CUCPX),d_plan->fw,fw_width*sizeof(CUCPX),
				     nf1*sizeof(CUCPX),nf2,cudaMemcpyDeviceToHost));

	return 0;
}

int copygpumem_to_cpumem_fk(cufinufft_plan *d_plan)
{
	int ms = d_plan->ms;
	int mt = d_plan->mt;
        checkCudaErrors(cudaMemcpy(d_plan->h_fk,d_plan->fk,ms*mt*sizeof(CUCPX),cudaMemcpyDeviceToHost));
        return 0;
}

void free_gpumemory(spread_opts opts, cufinufft_plan *d_plan)
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
	}
}
