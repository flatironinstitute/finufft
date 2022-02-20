#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>

#include <cufinufft_eitherprec.h>

#include "../contrib/utils.h"
#include "../src/common.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1;
	if (argc<2) {
		fprintf(stderr,
			"Usage: onedim_fseries_kernel_test nf1 [dim [tol [gpuversion [nf2 [nf3]]]]]\n"
			"Arguments:\n"
			"  nf1: The size of the upsampled fine grid size in x.\n"
			"  dim: Dimension of the nuFFT.\n"
			"  tol: NUFFT tolerance (default 1e-6).\n"
			"  gpuversion: Use gpu version or not (default True).\n"
			"  nf2: The size of the upsampled fine grid size in y. (default nf1)\n"
			"  nf3: The size of the upsampled fine grid size in z. (default nf3)\n"
			);
		return 1;
	}
	double w;
	sscanf(argv[1],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	int dim = 1;
	if (argc > 2) 
		sscanf(argv[2],"%d",&dim);
	FLT eps = 1e-6;
	if (argc > 3) 
		sscanf(argv[3],"%lf",&w); eps = (FLT)w;
	int gpu = 1;
	if (argc > 4) 
		sscanf(argv[4],"%d",&gpu);

	int nf2=nf1;
	if (argc > 5) 
		sscanf(argv[5],"%lf",&w); nf2 = (int)w;
	int nf3=nf1;
	if (argc > 6) 
		sscanf(argv[6],"%lf",&w); nf3 = (int)w;

	SPREAD_OPTS opts;
	FLT *fwkerhalf1, *fwkerhalf2, *fwkerhalf3;
	FLT *d_fwkerhalf1, *d_fwkerhalf2, *d_fwkerhalf3;
	checkCudaErrors(cudaMalloc(&d_fwkerhalf1, sizeof(FLT)*(nf1/2+1)));
	if(dim > 1)
		checkCudaErrors(cudaMalloc(&d_fwkerhalf2, sizeof(FLT)*(nf2/2+1)));
	if(dim > 2)
		checkCudaErrors(cudaMalloc(&d_fwkerhalf3, sizeof(FLT)*(nf3/2+1)));

	int ier = setup_spreader(opts, eps, 2.0, 0);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float milliseconds = 0;
	float gputime = 0;
	float cputime = 0;

	CNTime timer;
	if( !gpu ) {
		timer.start();
		fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
		if(dim > 1)
			fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
		if(dim > 2)
			fwkerhalf3 = (FLT*)malloc(sizeof(FLT)*(nf3/2+1));

		onedim_fseries_kernel(nf1, fwkerhalf1, opts);
		if(dim > 1)
			onedim_fseries_kernel(nf2, fwkerhalf2, opts);
		if(dim > 2)
			onedim_fseries_kernel(nf3, fwkerhalf3, opts);
		cputime = timer.elapsedsec();
		cudaEventRecord(start);
 		{
			checkCudaErrors(cudaMemcpy(d_fwkerhalf1,fwkerhalf1,
				sizeof(FLT)*(nf1/2+1),cudaMemcpyHostToDevice));
			if(dim > 1)
				checkCudaErrors(cudaMemcpy(d_fwkerhalf2,fwkerhalf2,
					sizeof(FLT)*(nf2/2+1),cudaMemcpyHostToDevice));
			if(dim > 2)
				checkCudaErrors(cudaMemcpy(d_fwkerhalf3,fwkerhalf3,
					sizeof(FLT)*(nf3/2+1),cudaMemcpyHostToDevice));
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		gputime = milliseconds;
		printf("[time  ] dim=%d, nf1=%8d, ns=%2d, CPU: %6.2f ms\n",
				dim, nf1, opts.nspread, gputime+cputime*1000);
		free(fwkerhalf1);
		if(dim > 1)
			free(fwkerhalf2);
		if(dim > 2)
			free(fwkerhalf3);
	} else {
		timer.start();
		complex<double> a[dim*MAX_NQUAD];
		FLT             f[dim*MAX_NQUAD];
		onedim_fseries_kernel_precomp(nf1, f, a, opts);
		if(dim > 1)
			onedim_fseries_kernel_precomp(nf2, f+MAX_NQUAD, a+MAX_NQUAD, opts);
		if(dim > 2)
			onedim_fseries_kernel_precomp(nf3, f+2*MAX_NQUAD, a+2*MAX_NQUAD, opts);
		cputime = timer.elapsedsec();

		cuDoubleComplex *d_a;
		FLT   *d_f;
		cudaEventRecord(start);
 		{
			checkCudaErrors(cudaMalloc(&d_a, dim*MAX_NQUAD*sizeof(cuDoubleComplex)));
			checkCudaErrors(cudaMalloc(&d_f, dim*MAX_NQUAD*sizeof(FLT)));
			checkCudaErrors(cudaMemcpy(d_a,a,
				dim*MAX_NQUAD*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_f,f,
				dim*MAX_NQUAD*sizeof(FLT),cudaMemcpyHostToDevice));
			ier = CUFSERIESKERNELCOMPUTE(dim, nf1, nf2, nf3, d_f, d_a, d_fwkerhalf1,
				d_fwkerhalf2, d_fwkerhalf3, opts.nspread);
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		gputime = milliseconds;
		printf("[time  ] dim=%d, nf1=%8d, ns=%2d, GPU: %6.2f ms\n",
				dim, nf1, opts.nspread, gputime+cputime*1000);
		cudaFree(d_a);
		cudaFree(d_f);
	}

#ifdef RESULT
	fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
	if(dim > 1)
		fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
	if(dim > 2)
		fwkerhalf3 = (FLT*)malloc(sizeof(FLT)*(nf3/2+1));

	checkCudaErrors(cudaMemcpy(fwkerhalf1,d_fwkerhalf1,sizeof(FLT)*(nf1/2+1),cudaMemcpyDeviceToHost));
	if(dim > 1)
		checkCudaErrors(cudaMemcpy(fwkerhalf2,d_fwkerhalf2,sizeof(FLT)*(nf2/2+1),cudaMemcpyDeviceToHost));
	if(dim > 2)
		checkCudaErrors(cudaMemcpy(fwkerhalf3,d_fwkerhalf3,sizeof(FLT)*(nf3/2+1),cudaMemcpyDeviceToHost));
	for(int i=0; i<nf1/2+1; i++)
		printf("%10.8e ", fwkerhalf1[i]);
	printf("\n");
	if(dim > 1)
		for(int i=0; i<nf2/2+1; i++)
			printf("%10.8e ", fwkerhalf2[i]);
		printf("\n");
	if(dim > 2)
		for(int i=0; i<nf3/2+1; i++)
			printf("%10.8e ", fwkerhalf3[i]);
		printf("\n");
#endif

	return 0;
}
