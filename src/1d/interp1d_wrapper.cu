#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "../cuspreadinterp.h"
#include "../memtransfer.h"
#include <profile.h>

using namespace std;

int CUFINUFFT_INTERP1D(int nf1, CUCPX* d_fw, int M, FLT *d_kx, CUCPX *d_c, 
	CUFINUFFT_PLAN d_plan)
/*
	This c function is written for only doing 1D interpolation. See 
	test/interp1d_test.cu for usage.

	note: not allocate,transfer and free memories on gpu.
	Melody Shih 11/21/21
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_plan->nf1 = nf1;
	d_plan->M = M;
	d_plan->maxbatchsize = 1;

	d_plan->kx = d_kx;
	d_plan->c  = d_c;
	d_plan->fw = d_fw;

	int ier;
	cudaEventRecord(start);
	ier = ALLOCGPUMEM1D_PLAN(d_plan);
	ier = ALLOCGPUMEM1D_NUPTS(d_plan);
	if(d_plan->opts.gpu_method == 1){
		ier = CUSPREAD1D_NUPTSDRIVEN_PROP(nf1,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread1d_subprob_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return ier;
		}
	}
	if(d_plan->opts.gpu_method == 2){
		ier = CUSPREAD1D_SUBPROB_PROP(nf1,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread1d_subprob_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return ier;
		}
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Obtain Interp Prop\t %.3g ms\n", d_plan->opts.gpu_method, 
		milliseconds);
#endif
	cudaEventRecord(start);
	ier = CUINTERP1D(d_plan,1);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Interp (%d)\t\t %.3g ms\n", d_plan->opts.gpu_method, 
		milliseconds);
#endif
	cudaEventRecord(start);
	FREEGPUMEMORY1D(d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	return ier;
}

int CUINTERP1D(CUFINUFFT_PLAN d_plan, int blksize)
/*
	A wrapper for different interpolation methods. 

	Methods available:
	(1) Non-uniform points driven
	(2) Subproblem

	Melody Shih 11/21/21
*/
{
	int nf1 = d_plan->nf1;
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
					ier = CUINTERP1D_NUPTSDRIVEN(nf1, M, d_plan, blksize);
					if(ier != 0 ){
						cout<<"error: cnufftspread1d_gpu_nuptsdriven"<<endl;
						return 1;
					}
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1"<<endl;
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

int CUINTERP1D_NUPTSDRIVEN(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize)
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
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 32;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;

	cudaEventRecord(start);
	if(d_plan->opts.gpu_kerevalmeth){
		for(int t=0; t<blksize; t++){
			Interp_1d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, 
				d_c+t*M, d_fw+t*nf1, M, ns, nf1, sigma, d_idxnupts, pirange);
		}
	}else{
		for(int t=0; t<blksize; t++){
			Interp_1d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, 
				d_c+t*M, d_fw+t*nf1, M, ns, nf1, es_c, es_beta, d_idxnupts, pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_1d_NUptsdriven (%d)\t%.3g ms\n", 
		milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}
