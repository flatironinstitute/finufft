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

int cufinufft2d_plan(int M, int ms, int mt, int iflag, const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	//ier=cufinufft_default_opts(opts,eps,upsampfac);
	int nf1 = (int) opts.upsampfac*ms;
	int nf2 = (int) opts.upsampfac*mt;
	int fftsign = (iflag>=0) ? 1 : -1;

	d_plan->ms = ms;
	d_plan->mt = mt;
	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->M = M;
	d_plan->iflag = fftsign;
#ifdef INFO
	printf("[info  ] 2d1: (ms,mt)=(%d,%d) (nf1, nf2)=(%d,%d) nj=%d\n", ms, mt, d_plan->nf1, d_plan->nf2, d_plan->M);
#endif

	// this may move to gpu
	CNTime timer; timer.start();
	FLT *fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
	FLT *fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
	onedim_fseries_kernel(nf1, fwkerhalf1, opts);
	onedim_fseries_kernel(nf2, fwkerhalf2, opts);
#ifdef TIME
	printf("[time  ] \tkernel fser (ns=%d):\t %.3g s\n", opts.nspread,timer.elapsedsec());
#endif

	cudaEventRecord(start);
	ier = allocgpumemory(opts, d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf1,fwkerhalf1,(nf1/2+1)*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf2,fwkerhalf2,(nf2/2+1)*sizeof(FLT),cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy fwkerhalf1,2 HtoD\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	cufftHandle fftplan;
	int ndata=1;
	int n[] = {nf2, nf1};
	int inembed[] = {nf2, d_plan->fw_width};
	cufftPlanMany(&fftplan,2,n,inembed,1,inembed[0]*inembed[1],inembed,1,inembed[0]*inembed[1],
			CUFFT_TYPE,ndata);
	d_plan->fftplan = fftplan;
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCUFFT Plan\t\t %.3g s\n", milliseconds/1000);
#endif
	return ier;
}

int cufinufft2d_setNUpts(FLT* h_kx, FLT* h_ky, const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Copy memory to device
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->kx,h_kx,d_plan->M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->ky,h_ky,d_plan->M*sizeof(FLT),cudaMemcpyHostToDevice));
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy kx,ky HtoD\t\t %.3g s\n", milliseconds/1000);
#endif
	return 0;
}

int cufinufft2d1_exec(CPX* h_c, CPX* h_fk, cufinufft_opts &opts, cufinufft_plan *d_plan)
{
	opts.spread_direction = 1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	// Copy memory to device
	//int ier = copycpumem_to_gpumem(opts, d_plan);
	checkCudaErrors(cudaMemcpy(d_plan->c,h_c,d_plan->M*sizeof(CUCPX),cudaMemcpyHostToDevice));
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy h_c HtoD\t\t %.3g s\n", milliseconds/1000);
#endif
	// Step 1: Spread
  cudaEventRecord(start);
  int ier = cuspread2d(opts, d_plan);
  if(ier != 0 ){
          printf("error: cuspread2d, method(%d)\n", opts.method);
          return 0;
  }
#ifdef TIME
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tSpread (%d)\t\t %.3g s\n", milliseconds/1000, opts.method);
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
	cudeconvolve2d(opts,d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tDeconvolve\t\t %.3g s\n", milliseconds/1000);
#endif
	cudaEventRecord(start);
        checkCudaErrors(cudaMemcpy(h_fk,d_plan->fk,d_plan->ms*d_plan->mt*sizeof(CUCPX),cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy fk memory DtoH \t %.3g s\n", milliseconds/1000);
#endif
	return ier;
}

int cufinufft2d2_exec(CPX* h_c, CPX* h_fk, cufinufft_opts &opts, cufinufft_plan *d_plan)
{
	opts.spread_direction = 2;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	// Copy memory to device
	//int ier = copycpumem_to_gpumem(opts, d_plan);
  checkCudaErrors(cudaMemcpy(d_plan->fk,h_fk,d_plan->ms*d_plan->mt*sizeof(CUCPX),cudaMemcpyHostToDevice));
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy h_fk HtoD\t\t %.3g s\n", milliseconds/1000);
#endif
	// Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
	cudaEventRecord(start);
	cudeconvolve2d(opts,d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAmplify & Copy fktofw\t %.3g s\n", milliseconds/1000);
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
  int ier = cuinterp2d(opts, d_plan);
  if(ier != 0 ){
          printf("error: cuinterp2d, method(%d)\n", opts.method);
          return 0;
  }
#ifdef TIME
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tUnspread (%d)\t\t %.3g s\n", milliseconds/1000,opts.method);
#endif

	cudaEventRecord(start);
  checkCudaErrors(cudaMemcpy(h_c,d_plan->c,d_plan->M*sizeof(CUCPX),cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy c DtoH\t\t %.3g s\n", milliseconds/1000);
#endif
	return ier;
}

int cufinufft2d_destroy(const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
        cufftDestroy(d_plan->fftplan);
        freegpumemory(opts, d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tFree gpu memory\t\t %.3g s\n", milliseconds/1000);
#endif
	return 0;
}

int cufinufft_default_opts(cufinufft_opts &opts,FLT eps,FLT upsampfac)
{
	// defaults... (user can change after this function called)
	opts.pirange = 1;             // user also should always set this
	opts.upsampfac = upsampfac;

	// for gpu
	opts.method = 5;
	opts.bin_size_x = 32;
	opts.bin_size_y = 32;
	opts.Horner = 0;
	opts.maxsubprobsize = 1000;
	opts.nthread_x = 16;
	opts.nthread_y = 16;

	// Set kernel width w (aka ns) and ES kernel beta parameter, in opts...
	int ns = std::ceil(-log10(eps/10.0));   // 1 digit per power of ten
	if (upsampfac!=2.0)           // override ns for custom sigma
		ns = std::ceil(-log(eps) / (PI*sqrt(1-1/upsampfac)));  // formula, gamma=1
	ns = max(2,ns);               // we don't have ns=1 version yet
	if (ns>MAX_NSPREAD) {         // clip to match allocated arrays
		fprintf(stderr,"setup_spreader: warning, kernel width ns=%d was clipped to max %d; will not match tolerance!\n",ns,MAX_NSPREAD);
		ns = MAX_NSPREAD;
	}
	opts.nspread = ns;
	opts.ES_halfwidth=(FLT)ns/2;   // constants to help ker eval (except Horner)
	opts.ES_c = 4.0/(FLT)(ns*ns);

	FLT betaoverns = 2.30;         // gives decent betas for default sigma=2.0
	if (ns==2) betaoverns = 2.20;  // some small-width tweaks...
	if (ns==3) betaoverns = 2.26;
	if (ns==4) betaoverns = 2.38;
	if (upsampfac!=2.0) {          // again, override beta for custom sigma
		FLT gamma=0.97;              // must match devel/gen_all_horner_C_code.m
		betaoverns = gamma*PI*(1-1/(2*upsampfac));  // formula based on cutoff
	}
	opts.ES_beta = betaoverns * (FLT)ns;    // set the kernel beta parameter
	//fprintf(stderr,"setup_spreader: sigma=%.6f, chose ns=%d beta=%.6f\n",(double)upsampfac,ns,(double)opts.ES_beta); // user hasn't set debug yet
	return 0;

}
