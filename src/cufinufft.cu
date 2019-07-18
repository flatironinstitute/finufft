#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include "spreadinterp.h"
#include "memtransfer.h"
#include "deconvolve.h"
#include "cufinufft.h"
#include "../finufft/utils.h"
#include "../finufft/common.h"

using namespace std;

int cufinufft_makeplan(finufft_type type, int dim, int *nmodes, int iflag, 
	int ntransf, FLT tol, int ntransfcufftplan, cufinufft_plan *d_plan)
{

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	assert(dim==2);
	int ier = setup_spreader_for_nufft(d_plan->spopts,tol,d_plan->opts);

	d_plan->ms = nmodes[0];
	d_plan->mt = nmodes[1];
	d_plan->mu = nmodes[2];

	int nf1=1, nf2=1, nf3=1;
	set_nf_type12(d_plan->ms, d_plan->opts, d_plan->spopts, &nf1);
	if(dim > 1)
		set_nf_type12(d_plan->mt, d_plan->opts, d_plan->spopts, &nf2); 
	if(dim > 2)
		set_nf_type12(d_plan->mu, d_plan->opts, d_plan->spopts, &nf3);
	int fftsign = (iflag>=0) ? 1 : -1;

	d_plan->nf1 = nf1;	
	d_plan->nf2 = nf2;	
	d_plan->nf3 = nf3;	
	d_plan->iflag = fftsign;
	d_plan->ntransf = ntransf;
	d_plan->ntransfcufftplan = ntransfcufftplan;
	d_plan->type = type;

	if(d_plan->type == type1)
		d_plan->spopts.spread_direction = 1; 
	if(d_plan->type == type2)
		d_plan->spopts.spread_direction = 2; 
	// this may move to gpu
	CNTime timer; timer.start();
	FLT *fwkerhalf1, *fwkerhalf2, *fwkerhalf3;

	fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
	onedim_fseries_kernel(nf1, fwkerhalf1, d_plan->spopts);
	if(dim > 1){	
		fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
		onedim_fseries_kernel(nf2, fwkerhalf2, d_plan->spopts);
	}
	if(dim > 2){	
		fwkerhalf3 = (FLT*)malloc(sizeof(FLT)*(nf3/2+1));
		onedim_fseries_kernel(nf3, fwkerhalf3, d_plan->spopts);
	}
#ifdef TIME
	printf("[time  ] \tkernel fser (ns=%d):\t %.3g s\n", d_plan->spopts.nspread,
		timer.elapsedsec());
#endif

	cudaEventRecord(start);
	ier = allocgpumem2d_plan(d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory plan %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf1,fwkerhalf1,(nf1/2+1)*
		sizeof(FLT),cudaMemcpyHostToDevice));
	if(dim > 1)
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf2,fwkerhalf2,(nf2/2+1)*
			sizeof(FLT),cudaMemcpyHostToDevice));
	if(dim > 2)
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf3,fwkerhalf3,(nf3/2+1)*
			sizeof(FLT),cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy fwkerhalf1,2 HtoD\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	cufftHandle fftplan;
	if(dim == 2){
		int n[] = {nf2, nf1};
		int inembed[] = {nf2, nf1};
		cufftPlanMany(&fftplan,dim,n,inembed,1,inembed[0]*inembed[1],inembed,1,
			inembed[0]*inembed[1],CUFFT_TYPE,ntransfcufftplan);
	}
	d_plan->fftplan = fftplan;
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCUFFT Plan\t\t %.3g s\n", milliseconds/1000);
#endif
	free(fwkerhalf1);
	if(dim > 1)
		free(fwkerhalf2);
	if(dim > 2)
		free(fwkerhalf3);
	return ier;
}
#if 0
int cufinufft_setNUpts(int M, FLT* h_kx, FLT* h_ky, cufinufft_plan *d_plan)
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;

	d_plan->M = M;
#ifdef INFO
	printf("[info  ] 2d1: (ms,mt)=(%d,%d) (nf1, nf2)=(%d,%d) nj=%d, ntransform = %d\n",
		d_plan->ms, d_plan->mt, d_plan->nf1, d_plan->nf2, d_plan->M, 
		d_plan->ntransf);
#endif
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int ier = allocgpumem2d_nupts(d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory NUpts%.3g s\n", milliseconds/1000);
#endif

	// Copy memory to device
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->kx,h_kx,d_plan->M*sizeof(FLT),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->ky,h_ky,d_plan->M*sizeof(FLT),
		cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy kx,ky HtoD\t\t %.3g s\n", milliseconds/1000);
#endif

	if(d_plan->opts.pirange == 1){
		cudaEventRecord(start);
		RescaleXY_2d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,d_plan->kx, 
			d_plan->ky);
		d_plan->opts.pirange = 0;
#ifdef SPREADTIME
	float milliseconds;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ]\tRescaleXY_2d\t\t %.3g ms\n", milliseconds);
#endif
	}
	if(d_plan->opts.gpu_method==5){
		ier = cuspread2d_subprob_prop(nf1,nf2,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread2d_subprob_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}
	if(d_plan->opts.gpu_method==6){
		int ier = cuspread2d_paul_prop(nf1,nf2,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread2d_paul_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}
	return 0;
}

int cufinufft_exec(CPX* h_c, CPX* h_fk, cufinufft_plan *d_plan)
{
	assert(d_plan->opts.spread_direction == 1);
	assert(d_plan->opts.pirange == 0);
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
		h_fkstart = h_fk + i*d_plan->ntransfcufftplan*d_plan->ms*d_plan->mt;
		checkCudaErrors(cudaMemcpy(d_plan->c,h_cstart,blksize*d_plan->M*
			sizeof(CUCPX),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_plan->fw,0,d_plan->ntransfcufftplan*
					d_plan->nf1*d_plan->nf2*sizeof(CUCPX)));// this is needed
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
		ier = cuspread2d(d_plan);
		if(ier != 0 ){
			printf("error: cuspread2d, method(%d)\n", d_plan->opts.gpu_method);
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
		cudeconvolve2d(d_plan);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tDeconvolve\t\t %.3g s\n", milliseconds/1000);
#endif
		cudaEventRecord(start);
		checkCudaErrors(cudaMemcpy(h_fkstart,d_plan->fk,blksize*d_plan->ms*
			d_plan->mt*sizeof(CUCPX),cudaMemcpyDeviceToHost));
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy fk memory DtoH \t %.3g s\n", milliseconds/1000);
#endif
	}
	return ier;
}

int cufinufft2d2_exec(CPX* h_c, CPX* h_fk, cufinufft_plan *d_plan)
{
	assert(d_plan->opts.spread_direction == 2);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int blksize, ier;
	CPX* h_fkstart;
	CPX* h_cstart;
	for(int i=0; i*d_plan->ntransfcufftplan < d_plan->ntransf; i++){
		blksize = min(d_plan->ntransf - i*d_plan->ntransfcufftplan, 
			d_plan->ntransfcufftplan);
		h_cstart  = h_c  + i*d_plan->ntransfcufftplan*d_plan->M;
		h_fkstart = h_fk + i*d_plan->ntransfcufftplan*d_plan->ms*d_plan->mt;

		checkCudaErrors(cudaMemcpy(d_plan->fk,h_fkstart,blksize*
			d_plan->ms*d_plan->mt*sizeof(CUCPX),cudaMemcpyHostToDevice));
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy h_fk HtoD\t\t %.3g s\n", milliseconds/1000);
#endif
		// Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
		cudaEventRecord(start);
		cudeconvolve2d(d_plan);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tAmplify & Copy fktofw\t %.3g s\n", milliseconds/1000);
#endif
		// Step 2: FFT
		cudaDeviceSynchronize();
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
		ier = cuinterp2d(d_plan);
		if(ier != 0 ){
			printf("error: cuinterp2d, method(%d)\n", d_plan->opts.gpu_method);
			return 0;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tUnspread (%d)\t\t %.3g s\n", milliseconds/1000,
			d_plan->opts.gpu_method);
#endif

		cudaDeviceSynchronize();
		cudaEventRecord(start);
		checkCudaErrors(cudaMemcpy(h_cstart,d_plan->c,blksize*d_plan->M*
			sizeof(CUCPX),cudaMemcpyDeviceToHost));
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy c DtoH\t\t %.3g s\n", milliseconds/1000);
#endif
	}
	return ier;
}
#endif

int cufinufft_destroy(cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cufftDestroy(d_plan->fftplan);
	freegpumemory2d(d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tFree gpu memory\t\t %.3g s\n", milliseconds/1000);
#endif
	return 0;
}

int cufinufft_default_opts(nufft_opts &opts)
{
	/* following options are for gpu */
	opts.gpu_nstreams = 16;
	opts.gpu_method = 5;
	opts.gpu_binsizex = 32;
	opts.gpu_binsizey = 32;
	opts.gpu_kerevalmeth = 1;
	opts.gpu_maxsubprobsize = 1000;

	opts.upsampfac = (FLT)2.0;   // sigma: either 2.0, or 1.25 for smaller RAM, FFTs

	/* following options are not used in cpu code */
	opts.chkbnds = -1;
	opts.debug = -1;
	opts.spread_debug = -1;
	opts.spread_sort = -1;        // use heuristic rule for whether to sort
	opts.spread_kerevalmeth = -1; // 0: direct exp(sqrt()), 1: Horner ppval
	opts.spread_kerpad = -1;      // (relevant iff kerevalmeth=0)

	return 0;
}
