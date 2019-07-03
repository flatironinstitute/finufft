#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "../spreadinterp.h"
#include "../memtransfer.h"

using namespace std;

// This function includes device memory allocation, transfer, free
int cufinufft_interp2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX *h_c, cufinufft_opts &opts, cufinufft_plan* d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;

	d_plan->ms = ms;
	d_plan->mt = mt;
	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->M = M;
	d_plan->ntransfcufftplan = 1;

	cudaEventRecord(start);
	ier = allocgpumemory2d(opts, d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Allocate GPU memory\t %.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	cudaMemcpy2D(d_plan->fw,d_plan->fw_width*sizeof(CUCPX),h_fw,nf1*sizeof(CUCPX),
			nf1*sizeof(CUCPX),nf2,cudaMemcpyHostToDevice);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory HtoD\t %.3g ms\n", milliseconds);
#endif
	if(opts.method == 5){
		ier = cuspread2d_subprob_prop(nf1,nf2,M,opts,d_plan);
		if(ier != 0 ){
			printf("error: cuspread2d_subprob_prop, method(%d)\n", opts.method);
			return 0;
		}
	}
	cudaEventRecord(start);
	ier = cuinterp2d(opts, d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Interp (%d)\t\t %.3g ms\n", opts.method, milliseconds);
#endif
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(h_c,d_plan->c,M*sizeof(CUCPX),cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory DtoH\t %.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	freegpumemory2d(opts, d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	return ier;
}

// a wrapper of different methods of spreader
int cuinterp2d(cufinufft_opts &opts, cufinufft_plan* d_plan)
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int fw_width = d_plan->fw_width;
	int M = d_plan->M;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	switch(opts.method)
	{
		case 1:
			{
				cudaEventRecord(start);
				ier = cuinterp2d_idriven(nf1, nf2, fw_width, M, opts, d_plan);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_idriven"<<endl;
					return 1;
				}
			}
			break;
		case 5:
			{
				cudaEventRecord(start);
				ier = cuinterp2d_subprob(nf1, nf2, fw_width, M, opts, d_plan);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_hybrid"<<endl;
					return 1;
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1 or 5"<<endl;
			return 2;
	}
#ifdef SPREADTIME
	float milliseconds;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Interp " << milliseconds <<" ms"<<endl;
#endif
	return ier;
}

int cuinterp2d_idriven(int nf1, int nf2, int fw_width, int M, const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	cudaEventRecord(start);

	for(int t=0; t<d_plan->ntransfcufftplan; t++){
		Interp_2d_Idriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, 
				M, ns, nf1, nf2, es_c, es_beta, 
				fw_width);
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_2d_Idriven \t%.3g ms\n", milliseconds);
#endif
	return 0;
}

int cuinterp2d_subprob(int nf1, int nf2, int fw_width, int M, const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;
	int maxsubprobsize=opts.maxsubprobsize;

	// assume that bin_size_x > ns/2;
	int bin_size_x=opts.bin_size_x;
	int bin_size_y=opts.bin_size_y;
	int numbins[2];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<opts.bin_size_x<<"x"<<opts.bin_size_y<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;
	int *d_subprob_to_bin = d_plan->subprob_to_bin;
	int totalnumsubprob=d_plan->totalnumsubprob;

	FLT sigma=opts.upsampfac;
	cudaEventRecord(start);
	size_t sharedplanorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*sizeof(CUCPX);
	if(sharedplanorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	for(int t=0; t<d_plan->ntransfcufftplan; t++){
		Interp_2d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
				d_kx, d_ky, d_c+t*M,
				d_fw+t*nf1*nf2, M, ns, nf1, nf2,
				es_c, es_beta, sigma, fw_width,
				d_binstartpts, d_binsize,
				bin_size_x, bin_size_y,
				d_subprob_to_bin, d_subprobstartpts,
				d_numsubprob, maxsubprobsize,
				numbins[0], numbins[1], d_idxnupts);
	}
#ifdef SPREADTIME
 	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_2d_Subprob \t\t%.3g ms\n", milliseconds);
#endif
	return 0;
}
