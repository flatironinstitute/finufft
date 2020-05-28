#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include <cufinufft.h>
#include "../cuspreadinterp.h"
#include "../cudeconvolve.h"
#include "../memtransfer.h"

using namespace std;

int cufinufft3d1_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan)
/*  
	3D Type-1 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: spread data to oversampled regular mesh using kernel
		Step 2: compute FFT on uniform mesh
		Step 3: deconvolve by division of each Fourier mode independently by the
		        Fourier series coefficient of the kernel.

	Melody Shih 07/25/19		
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int blksize; 
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for(int i=0; i*d_plan->maxbatchsize < d_plan->ntransf; i++){
		blksize = min(d_plan->ntransf - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart = d_c + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*d_plan->ms*d_plan->mt*
			d_plan->mu;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;

		checkCudaErrors(cudaMemset(d_plan->fw,0,d_plan->maxbatchsize*
					d_plan->nf1*d_plan->nf2*d_plan->nf3*sizeof(CUCPX)));
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tInitialize fw\t\t %.3g s\n", milliseconds/1000);
#endif
		// Step 1: Spread
		cudaEventRecord(start);
		ier = cuspread3d(d_plan, blksize);
		if(ier != 0 ){
			printf("error: cuspread3d, method(%d)\n", d_plan->opts.gpu_method);
			return 0;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tSpread (%d)\t\t %.3g s\n", milliseconds/1000, 
			d_plan->opts.gpu_method);
#endif
		// Step 2: FFT
		cudaEventRecord(start);
		CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCUFFT Exec\t\t %.3g s\n", milliseconds/1000);
#endif

		// Step 3: deconvolve and shuffle
		cudaEventRecord(start);
		cudeconvolve3d(d_plan, blksize);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tDeconvolve\t\t %.3g s\n", milliseconds/1000);
#endif
	}
	return ier;
}

int cufinufft3d2_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan)
/*  
	3D Type-2 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel 
		        Fourier coeff
		Step 2: compute FFT on uniform mesh
		Step 3: interpolate data to regular mesh

	Melody Shih 07/25/19		
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int blksize;
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for(int i=0; i*d_plan->maxbatchsize < d_plan->ntransf; i++){
		blksize = min(d_plan->ntransf - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart  = d_c  + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*d_plan->ms*d_plan->mt*
			d_plan->mu;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;

		// Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
		cudaEventRecord(start);
		cudeconvolve3d(d_plan, blksize);
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tAmplify & Copy fktofw\t %.3g s\n", milliseconds/1000);
#endif
		// Step 2: FFT
		cudaEventRecord(start);
		cudaDeviceSynchronize();
		CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCUFFT Exec\t\t %.3g s\n", milliseconds/1000);
#endif

		// Step 3: deconvolve and shuffle
		cudaEventRecord(start);
		ier = cuinterp3d(d_plan, blksize);
		if(ier != 0 ){
			printf("error: cuinterp3d, method(%d)\n", d_plan->opts.gpu_method);
			return 0;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tUnspread (%d)\t\t %.3g s\n", milliseconds/1000,
			d_plan->opts.gpu_method);
#endif

		cudaEventRecord(start);
#if 0
		if(d_plan->nstreams != 1)
			cudaDeviceSynchronize();
#endif
	}
	return ier;
}

#if 0
int cufinufft3d_plan(int M, int ms, int mt, int mu, int ntransf, 
	int maxbatchsize, int iflag, const cufinufft_opts opts, 
	cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	//ier=cufinufft_default_opts(opts,eps,upsampfac);
	int nf1 = (int) d_plan->opts.gpu_upsampfac*ms;
	int nf2 = (int) d_plan->opts.gpu_upsampfac*mt;
	int nf3 = (int) d_plan->opts.gpu_upsampfac*mu;
	int fftsign = (iflag>=0) ? 1 : -1;

	d_plan->ms = ms;
	d_plan->mt = mt;
	d_plan->mu = mu;
	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->nf3 = nf3;
	d_plan->M = M;
	d_plan->iflag = fftsign;
	d_plan->ntransf = ntransf;
	d_plan->maxbatchsize = maxbatchsize;
#ifdef INFO
	printf("[info  ] 3d: (ms,mt,mu)=(%d,%d) (nf1, nf2, nf3)=(%d,%d,%d) nj=%d, ntransform = %d\n",
		ms, mt, mu, d_plan->nf1, d_plan->nf2, d_plan->nf3, d_plan->M, 
		d_plan->ntransf);
#endif

	// this may move to gpu
	CNTime timer; timer.start();
	FLT *fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
	FLT *fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
	FLT *fwkerhalf3 = (FLT*)malloc(sizeof(FLT)*(nf3/2+1));
	onedim_fseries_kernel(nf1, fwkerhalf1, opts);
	onedim_fseries_kernel(nf2, fwkerhalf2, opts);
	onedim_fseries_kernel(nf3, fwkerhalf3, opts);
#ifdef DEBUG
	printf("[time  ] \tkernel fser (ns=%d):\t %.3g s\n", d_plan->opts.gpu_nspread,timer.elapsedsec());
#endif

	cudaEventRecord(start);
	ier = allocgpumemory3d(opts, d_plan);
#ifdef DEBUG
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf1,fwkerhalf1,(nf1/2+1)*
		sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf2,fwkerhalf2,(nf2/2+1)*
		sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf3,fwkerhalf3,(nf3/2+1)*
		sizeof(FLT),cudaMemcpyHostToDevice));
#ifdef DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy fwkerhalf1,2,3 HtoD %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	cufftHandle fftplan;
	int dim = 3;
	int n[] = {nf3, nf2, nf1};
	int inembed[] = {nf3, nf2, nf1};
	int istride = 1;
	cufftPlanMany(&fftplan,dim,n,inembed,istride,inembed[0]*inembed[1]*inembed[2],
		inembed,istride,inembed[0]*inembed[1]*inembed[2],CUFFT_TYPE,
		maxbatchsize);
	d_plan->fftplan = fftplan;
#ifdef DEBUG
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCUFFT Plan\t\t %.3g s\n", milliseconds/1000);
#endif
	return ier;
}
int cufinufft3d_setNUpts(FLT* h_kx, FLT* h_ky, FLT *h_kz, cufinufft_opts &opts, cufinufft_plan *d_plan)
{
	int M = d_plan->M;
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Copy memory to device
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->kx,h_kx,d_plan->M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->ky,h_ky,d_plan->M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->kz,h_kz,d_plan->M*sizeof(FLT),cudaMemcpyHostToDevice));
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy kx,ky,kz HtoD\t %.3g s\n", milliseconds/1000);
#endif

	if(d_plan->opts.gpu_pirange==1){
		cudaEventRecord(start);
		RescaleXY_3d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,nf3,d_plan->kx,
			d_plan->ky,d_plan->kz);
		d_plan->opts.gpu_pirange=0;
#ifdef SPREADTIME
		float milliseconds;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ]\tRescaleXY_3d\t\t %.3g ms\n", milliseconds);
#endif
	}

	cudaEventRecord(start);
	if(d_plan->opts.gpu_method == 5){
		int ier = cuspread3d_subprob_prop(nf1,nf2,nf3,M,opts,d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d_subprob_prop, method(%d)\n", d_plan->opts.gpu_method);
			return 0;
		}
	}
	if(d_plan->opts.gpu_method == 1 || d_plan->opts.gpu_method ==  2 || d_plan->opts.gpu_method == 3){
		int ier = cuspread3d_blockgather_prop(nf1,nf2,nf3,M,opts,d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d_blockgather_prop, method(%d)\n", d_plan->opts.gpu_method);
			return 0;
		}
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tSetup Subprob properties %.3g s\n", 
		milliseconds/1000);
#endif
	return 0;
}
#endif
