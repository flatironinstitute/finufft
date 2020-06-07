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
