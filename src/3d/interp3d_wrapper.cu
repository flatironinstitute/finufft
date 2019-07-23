#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "../spreadinterp.h"
#include "../memtransfer.h"
#include "../profile.h"

using namespace std;

// This function includes device memory allocation, transfer, free
int cufinufft_interp3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, 
	CPX* h_fw, int M, FLT *h_kx, FLT *h_ky, FLT *h_kz, CPX *h_c, 
	cufinufft_opts &opts, cufinufft_plan* d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;

	d_plan->ms = ms;
	d_plan->mt = mt;
	d_plan->mu = mu;
	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->nf3 = nf3;
	d_plan->M = M;
	d_plan->ntransfcufftplan = 1;

	cudaEventRecord(start);
	ier = allocgpumemory3d(opts, d_plan);
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
	checkCudaErrors(cudaMemcpy(d_plan->kz,h_kz,M*sizeof(FLT),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->fw,h_fw,nf1*nf2*nf3*sizeof(CUCPX),
		cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory HtoD\t %.3g ms\n", milliseconds);
#endif
	if(opts.method == 5){
		ier = cuspread3d_subprob_prop(nf1,nf2,nf3,M,opts,d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d_subprob_prop, method(%d)\n", opts.method);
			return 0;
		}
	}
	cudaEventRecord(start);
	ier = cuinterp3d(opts, d_plan);
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
	freegpumemory3d(opts, d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	return ier;
}

// a wrapper of different methods of spreader
int cuinterp3d(cufinufft_opts &opts, cufinufft_plan* d_plan)
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int M = d_plan->M;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	switch(opts.method)
	{
		case 4:
			{
				cudaEventRecord(start);
				{
					PROFILE_CUDA_GROUP("Interpolation", 6);
					ier = cuinterp3d_idriven(nf1, nf2, nf3, M, opts, d_plan);
					if(ier != 0 ){
						cout<<"error: cnufftspread3d_gpu_idriven"<<endl;
						return 1;
					}
				}
			}
			break;
		case 5:
			{
				cudaEventRecord(start);
				{
					PROFILE_CUDA_GROUP("Interpolation", 6);
					ier = cuinterp3d_subprob(nf1, nf2, nf3, M, opts, d_plan);
					if(ier != 0 ){
						cout<<"error: cnufftspread3d_gpu_subprob"<<endl;
						return 1;
					}
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 4"<<endl;
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

int cuinterp3d_idriven(int nf1, int nf2, int nf3, int M, 
	const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;
	FLT sigma=opts.upsampfac;

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	FLT* d_kz = d_plan->kz;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;

	cudaEventRecord(start);
	if(opts.Horner){
#if 0
		cudaStream_t *streams = d_plan->streams;
		int nstreams = d_plan->nstreams;
		for(int t=0; t<d_plan->ntransfcufftplan; t++){
			Interp_3d_Idriven_Horner<<<blocks, threadsPerBlock, 0, 
				streams[t%nstreams]>>>(d_kx, d_ky, d_kz, d_c+t*M, 
				d_fw+t*nf1*nf2*nf3, M, ns, nf1, nf2, nf3, sigma);
		}
#else 
		for(int t=0; t<d_plan->ntransfcufftplan; t++){
			Interp_3d_Idriven_Horner<<<blocks, threadsPerBlock, 0, 
				0>>>(d_kx, d_ky, d_kz, d_c+t*M, 
				d_fw+t*nf1*nf2*nf3, M, ns, nf1, nf2, nf3, sigma);
		}
#endif
#ifdef SPREADTIME
			float milliseconds = 0;
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("[time  ] \tKernel Interp_3d_Idriven_Horner \t%.3g ms\n", milliseconds);
#endif
	}else{
#if 0
		cudaStream_t *streams = d_plan->streams;
		int nstreams = d_plan->nstreams;
		for(int t=0; t<d_plan->ntransfcufftplan; t++){
			Interp_3d_Idriven<<<blocks, threadsPerBlock, 0, streams[t%nstreams]
				>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, M, ns, 
				nf1, nf2, nf3,es_c, es_beta);
		}
#else
		for(int t=0; t<d_plan->ntransfcufftplan; t++){
			Interp_3d_Idriven<<<blocks, threadsPerBlock, 0, 0 
				>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, M, ns, 
				nf1, nf2, nf3,es_c, es_beta);
		}
#endif
#ifdef SPREADTIME
			float milliseconds = 0;
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("[time  ] \tKernel Interp_3d_Idriven \t%.3g ms\n", milliseconds);
#endif
	}
	return 0;
}

int cuinterp3d_subprob(int nf1, int nf2, int nf3, int M, const cufinufft_opts opts, 
	cufinufft_plan *d_plan)
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
	int bin_size_z=opts.bin_size_z;
	int numbins[3];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
	numbins[2] = ceil((FLT) nf3/bin_size_z);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<opts.bin_size_x<<"x"<<opts.bin_size_y<<"x"<<opts.bin_size_z<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"x"<<numbins[2]
	<<"]"<<endl;
#endif

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	FLT* d_kz = d_plan->kz;
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
	size_t sharedplanorysize = (bin_size_x+2*ceil(ns/2.0))*
		(bin_size_y+2*ceil(ns/2.0))*(bin_size_z+2*ceil(ns/2.0))*sizeof(CUCPX);
	if(sharedplanorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	for(int t=0; t<d_plan->ntransfcufftplan; t++){
		Interp_3d_Subprob_Horner<<<totalnumsubprob, 256,
			sharedplanorysize>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, 
			M, ns, nf1, nf2, nf3, sigma, d_binstartpts, d_binsize, bin_size_x,
			bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts,
			d_numsubprob, maxsubprobsize,numbins[0], numbins[1], numbins[2],
			d_idxnupts);
#ifdef SPREADTIME
			float milliseconds = 0;
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("[time  ] \tKernel Interp_3d_Subprob_Horner \t%.3g ms\n", milliseconds);
#endif
	}
	return 0;
}
