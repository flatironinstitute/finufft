#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include "../spreadinterp.h"
#include "../memtransfer.h"
#include "../deconvolve.h"
#include "../cufinufft.h"
#include "../../finufft/utils.h"
#include "../../finufft/common.h"

using namespace std;

int cufinufft3d_plan(int M, int ms, int mt, int mu, int ntransf, 
	int ntransfcufftplan, int iflag, const cufinufft_opts opts, 
	cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	//ier=cufinufft_default_opts(opts,eps,upsampfac);
	int nf1 = (int) opts.upsampfac*ms;
	int nf2 = (int) opts.upsampfac*mt;
	int nf3 = (int) opts.upsampfac*mu;
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
	d_plan->ntransfcufftplan = ntransfcufftplan;
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
	printf("[time  ] \tkernel fser (ns=%d):\t %.3g s\n", opts.nspread,timer.elapsedsec());
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
		ntransfcufftplan);
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

	if(opts.pirange==1){
		cudaEventRecord(start);
		RescaleXY_3d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,nf3,d_plan->kx,
			d_plan->ky,d_plan->kz);
		opts.pirange=0;
#ifdef SPREADTIME
		float milliseconds;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ]\tRescaleXY_3d\t\t %.3g ms\n", milliseconds);
#endif
	}

	cudaEventRecord(start);
	if(opts.method == 5){
		int ier = cuspread3d_subprob_prop(nf1,nf2,nf3,M,opts,d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d_subprob_prop, method(%d)\n", opts.method);
			return 0;
		}
	}
	if(opts.method == 1 || opts.method ==  2 || opts.method == 3){
		int ier = cuspread3d_gather_prop(nf1,nf2,nf3,M,opts,d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d_gather_prop, method(%d)\n", opts.method);
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

int cufinufft3d1_exec(CPX* h_c, CPX* h_fk, cufinufft_opts &opts, cufinufft_plan *d_plan)
{
	opts.spread_direction = 1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//int ntransfpertime = d_plan->ntransfpertime;
	cudaEventRecord(start);
	// Copy memory to device
	int blksize, ier;
	CPX* h_fkstart;
	CPX* h_cstart;
	for(int i=0; i*d_plan->ntransfcufftplan < d_plan->ntransf; i++){
		blksize = min(d_plan->ntransf - i*d_plan->ntransfcufftplan, 
		d_plan->ntransfcufftplan);
		h_cstart = h_c + i*d_plan->ntransfcufftplan*d_plan->M;
		h_fkstart = h_fk + i*d_plan->ntransfcufftplan*d_plan->ms*d_plan->mt*
			d_plan->mu;

		checkCudaErrors(cudaMemcpy(d_plan->c,h_cstart,blksize*d_plan->M*
			sizeof(CUCPX),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_plan->fw,0,d_plan->ntransfcufftplan*
					d_plan->nf1*d_plan->nf2*d_plan->nf3*sizeof(CUCPX)));
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy h_c HtoD, Initialize fw to 0\t\t %.3g s\n", 
			milliseconds/1000);
#endif
		// Step 1: Spread
		cudaEventRecord(start);
		ier = cuspread3d(opts, d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d, method(%d)\n", opts.method);
			return 0;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tSpread (%d)\t\t %.3g s\n", milliseconds/1000, 
			opts.method);
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
		cudeconvolve3d(opts,d_plan);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tDeconvolve\t\t %.3g s\n", milliseconds/1000);
#endif
		cudaEventRecord(start);
		checkCudaErrors(cudaMemcpy(h_fkstart,d_plan->fk,blksize*d_plan->ms*
			d_plan->mt*d_plan->mu*sizeof(CUCPX),cudaMemcpyDeviceToHost));
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy fk memory DtoH \t %.3g s\n", milliseconds/1000);
#endif
	}
	return ier;
}

int cufinufft3d2_exec(CPX* h_c, CPX* h_fk, cufinufft_opts &opts, 
	cufinufft_plan *d_plan)
{
	opts.spread_direction = 2;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	int blksize, ier;
	CPX* h_fkstart;
	CPX* h_cstart;
	for(int i=0; i*d_plan->ntransfcufftplan < d_plan->ntransf; i++){
		blksize = min(d_plan->ntransf - i*d_plan->ntransfcufftplan, 
			d_plan->ntransfcufftplan);
		h_cstart  = h_c  + i*d_plan->ntransfcufftplan*d_plan->M;
		h_fkstart = h_fk + i*d_plan->ntransfcufftplan*d_plan->ms*d_plan->mt*
			d_plan->mu;

		cudaEventRecord(start);
		checkCudaErrors(cudaMemcpy(d_plan->fk,h_fkstart,blksize*
			d_plan->ms*d_plan->mt*d_plan->mu*sizeof(CUCPX),cudaMemcpyHostToDevice));
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy h_fk HtoD\t\t %.3g s\n", milliseconds/1000);
#endif
		// Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
		cudaEventRecord(start);
		cudeconvolve3d(opts,d_plan);
#ifdef TIME
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
		ier = cuinterp3d(opts, d_plan);
		if(ier != 0 ){
			printf("error: cuinterp3d, method(%d)\n", opts.method);
			return 0;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tUnspread (%d)\t\t %.3g s\n", milliseconds/1000,opts.method);
#endif

		cudaEventRecord(start);
#if 0
		if(d_plan->nstreams != 1)
			cudaDeviceSynchronize();
#endif
		checkCudaErrors(cudaMemcpy(h_cstart,d_plan->c,blksize*d_plan->M*sizeof(CUCPX),cudaMemcpyDeviceToHost));
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy c DtoH\t\t %.3g s\n", milliseconds/1000);
#endif
	}
	return ier;
}

int cufinufft3d_destroy(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cufftDestroy(d_plan->fftplan);
	freegpumemory3d(opts, d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tFree gpu memory\t\t %.3g s\n", milliseconds/1000);
#endif
	return 0;
}
