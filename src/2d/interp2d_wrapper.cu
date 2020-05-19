#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// cub library
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "../cuspreadinterp.h"
#include "../memtransfer.h"
#include "../profile.h"

using namespace std;

int cufinufft_interp2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, 
	FLT *h_kx, FLT *h_ky, CPX *h_c, cufinufft_plan* d_plan)
/*
	This c function is written for only doing 2D interpolation. It includes 
	allocating, transfering and freeing the memories on gpu. See 
	test/interp_2d.cu for usage.

	Melody Shih 07/25/19
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_plan->ms = ms;
	d_plan->mt = mt;
	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->M = M;
	d_plan->ntransfcufftplan = 1;

	int ier;
	//int ier = setup_spreader_for_nufft(d_plan->spopts, eps, d_plan->opts);
	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->c,M*sizeof(CUCPX)));

	cudaEventRecord(start);
	ier = allocgpumem2d_plan(d_plan);
	ier = allocgpumem2d_nupts(d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Allocate GPU memory\t %.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->kx,h_kx,M*sizeof(FLT),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->ky,h_ky,M*sizeof(FLT),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->fw,h_fw,nf1*nf2*sizeof(CUCPX),
		cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory HtoD\t %.3g ms\n", milliseconds);
#endif
	if(d_plan->opts.gpu_method == 1){
		ier = cuspread2d_nuptsdriven_prop(nf1,nf2,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread2d_subprob_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}
	if(d_plan->opts.gpu_method == 2){
		ier = cuspread2d_subprob_prop(nf1,nf2,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread2d_subprob_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}
	cudaEventRecord(start);
	ier = cuinterp2d(d_plan,1);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Interp (%d)\t\t %.3g ms\n", d_plan->opts.gpu_method, 
		milliseconds);
#endif
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(h_c,d_plan->c,M*sizeof(CUCPX),
		cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory DtoH\t %.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	freegpumemory2d(d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	cudaFree(d_plan->kx);
	cudaFree(d_plan->ky);
	cudaFree(d_plan->c);
	return ier;
}

int cuinterp2d(cufinufft_plan* d_plan, int blksize)
/*
	A wrapper for different interpolation methods. 

	Methods available:
	(1) Non-uniform points driven
	(2) Subproblem

	Melody Shih 07/25/19
*/
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int M = d_plan->M;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				cudaEventRecord(start);
				{
					PROFILE_CUDA_GROUP("Spreading", 6);
					ier = cuinterp2d_nuptsdriven(nf1, nf2, M, d_plan, blksize);
					if(ier != 0 ){
						cout<<"error: cnufftspread2d_gpu_nuptsdriven"<<endl;
						return 1;
					}
				}
			}
			break;
		case 2:
			{
				cudaEventRecord(start);
				ier = cuinterp2d_subprob(nf1, nf2, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cuinterp2d_subprob"<<endl;
					return 1;
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1 or 2"<<endl;
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

int cuinterp2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan *d_plan,
	int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=d_plan->spopts.nspread;   // psi's support in terms of number of cells
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	FLT sigma=d_plan->opts.upsampfac;
	int pirange=d_plan->spopts.pirange;
	int *d_idxnupts=d_plan->idxnupts;

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 32;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;

	cudaEventRecord(start);
	if(d_plan->opts.gpu_kerevalmeth){
#if 0
		cudaStream_t *streams = d_plan->streams;
		int nstreams = d_plan->opts.gpu_nstreams;
		for(int t=0; t<d_plan->ntransfcufftplan; t++){
			Interp_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock, 0, 
				streams[t%nstreams]>>>(d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, 
				ns, nf1, nf2, sigma);
		}
#else 
		for(int t=0; t<blksize; t++){
			Interp_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, 
				d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, sigma, 
				d_idxnupts, pirange);
		}
#endif
	}else{
#if 0
		cudaStream_t *streams = d_plan->streams;
		int nstreams = d_plan->opts.gpu_nstreams;
		for(int t=0; t<d_plan->ntransfcufftplan; t++){
			Interp_2d_NUptsdriven<<<blocks, threadsPerBlock, 0, streams[t%nstreams]
				>>>(d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, es_c, 
				es_beta);
		}
#else
		for(int t=0; t<blksize; t++){
			Interp_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, 
				d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, es_c, es_beta, 
				d_idxnupts, pirange);
		}
#endif
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_2d_NUptsdriven (%d)\t%.3g ms\n", 
		milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}

int cuinterp2d_subprob(int nf1, int nf2, int M, cufinufft_plan *d_plan,
	int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=d_plan->spopts.nspread;   // psi's support in terms of number of cells
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;

	// assume that bin_size_x > ns/2;
	int bin_size_x=d_plan->opts.gpu_binsizex;
	int bin_size_y=d_plan->opts.gpu_binsizey;
	int numbins[2];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"]"<<endl;
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
	int pirange=d_plan->spopts.pirange;

	FLT sigma=d_plan->opts.upsampfac;
	cudaEventRecord(start);
	size_t sharedplanorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*
		ceil(ns/2.0))*sizeof(CUCPX);
	if(sharedplanorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	if(d_plan->opts.gpu_kerevalmeth){
		for(int t=0; t<blksize; t++){
			Interp_2d_Subprob_Horner<<<totalnumsubprob, 256, sharedplanorysize>>>(
					d_kx, d_ky, d_c+t*M,
					d_fw+t*nf1*nf2, M, ns, nf1, nf2, sigma,
					d_binstartpts, d_binsize,
					bin_size_x, bin_size_y,
					d_subprob_to_bin, d_subprobstartpts,
					d_numsubprob, maxsubprobsize,
					numbins[0], numbins[1], d_idxnupts, pirange);
		}
	}else{
		for(int t=0; t<blksize; t++){
			Interp_2d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
					d_kx, d_ky, d_c+t*M,
					d_fw+t*nf1*nf2, M, ns, nf1, nf2,
					es_c, es_beta, sigma,
					d_binstartpts, d_binsize,
					bin_size_x, bin_size_y,
					d_subprob_to_bin, d_subprobstartpts,
					d_numsubprob, maxsubprobsize,
					numbins[0], numbins[1], d_idxnupts, pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_2d_Subprob (%d)\t\t%.3g ms\n", 
		milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}
