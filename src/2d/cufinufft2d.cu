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

using namespace std;

int cufinufft2d1_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan)
/*  
	2D Type-1 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: spread data to oversampled regular mesh using kernel
		Step 2: compute FFT on uniform mesh
		Step 3: deconvolve by division of each Fourier mode independently by the
		        Fourier series coefficient of the kernel.

	Melody Shih 07/25/19		
*/
{
	assert(d_plan->spopts.spread_direction == 1);
	//assert(d_plan->spopts.pirange == 0);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//int ntransfpertime = d_plan->ntransfpertime;
	cudaEventRecord(start);
	// Copy memory to device
	//int blksize;
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for(int i=0; i*d_plan->ntransfcufftplan < d_plan->ntransf; i++){
		//blksize = min(d_plan->ntransf - i*d_plan->ntransfcufftplan, 
		//	d_plan->ntransfcufftplan);
		d_cstart = d_c + i*d_plan->ntransfcufftplan*d_plan->M;
		d_fkstart = d_fk + i*d_plan->ntransfcufftplan*d_plan->ms*d_plan->mt;
		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;
#if 0
		checkCudaErrors(cudaMemcpy(d_plan->c,h_cstart,blksize*d_plan->M*
			sizeof(CUCPX),cudaMemcpyHostToDevice));
#endif
		checkCudaErrors(cudaMemset(d_plan->fw,0,d_plan->ntransfcufftplan*
					d_plan->nf1*d_plan->nf2*sizeof(CUCPX)));// this is needed
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tInitialize fw to 0\t %.3g s\n", 
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
#if 0
		cudaEventRecord(start);
		checkCudaErrors(cudaMemcpy(h_fkstart,d_plan->fk,blksize*d_plan->ms*
			d_plan->mt*sizeof(CUCPX),cudaMemcpyDeviceToHost));
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy fk memory DtoH \t %.3g s\n", milliseconds/1000);
#endif
#endif
	}
	return ier;
}

int cufinufft2d2_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan)
/*  
	2D Type-2 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel 
		        Fourier coeff
		Step 2: compute FFT on uniform mesh
		Step 3: interpolate data to regular mesh

	Melody Shih 07/25/19
*/
{
	assert(d_plan->spopts.spread_direction == 2);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	//int blksize;
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for(int i=0; i*d_plan->ntransfcufftplan < d_plan->ntransf; i++){
		//blksize = min(d_plan->ntransf - i*d_plan->ntransfcufftplan, 
		//	d_plan->ntransfcufftplan);
		d_cstart  = d_c  + i*d_plan->ntransfcufftplan*d_plan->M;
		d_fkstart = d_fk + i*d_plan->ntransfcufftplan*d_plan->ms*d_plan->mt;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;
#if 0
		checkCudaErrors(cudaMemcpy(d_plan->fk,h_fkstart,blksize*
			d_plan->ms*d_plan->mt*sizeof(CUCPX),cudaMemcpyHostToDevice));
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCopy h_fk HtoD\t\t %.3g s\n", milliseconds/1000);
#endif
#endif
		// Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
		cudaEventRecord(start);
		cudeconvolve2d(d_plan);
#ifdef TIME
		float milliseconds = 0;
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
#if 0
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
#endif
	}
	return ier;
}

