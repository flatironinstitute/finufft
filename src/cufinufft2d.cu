#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include "spread.h"
#include "cufinufft.h"
#include "finufft/utils.h"

using namespace std;

int cufinufft2d(int M, FLT* h_kx, FLT* h_ky, CPX* h_c, FLT tol, 
                int iflag, int nf1, int nf2, CPX* h_fw, spread_opts opts)
{
	cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
	int ier;
	// Step 0: Allocate and transfer memory for GPU
#ifdef INFO
	cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
#endif
	
        int fw_width;
        FLT *d_kx,*d_ky;
        gpuComplex *d_c,*d_fw;

        cudaEventRecord(start);
        ier = cnufft_allocgpumemory(nf1, nf2, M, &fw_width, h_fw, &d_fw, h_kx, &d_kx,
                                    h_ky, &d_ky, h_c, &d_c);
#ifdef TIME
	float milliseconds;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Allocate GPU memory\t %.3g ms\n", milliseconds);
#endif

        cudaEventRecord(start);
        ier = cnufft_copycpumem_to_gpumem(nf1, nf2, M, fw_width, h_fw, d_fw, h_kx, d_kx,
                                          h_ky, d_ky, h_c, d_c);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory HtoD\t %.3g ms\n", milliseconds);
#endif

	// Step 1: Spread
        cudaEventRecord(start);
	ier = cnufftspread2d_gpu_idriven(nf1, nf2, fw_width, d_fw, M, d_kx, d_ky, d_c, opts);
	if(ier != 0 ){
		cout<<"error: cnufftspread2d_gpu_idriven"<<endl;
		return 0;
	}
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Spread\t\t\t %.3g ms\n", milliseconds);
#endif
	// Step 2: Call FFT
	cufftHandle plan;
	int n[] = {nf2, nf1};
	int inembed[] = {nf2, fw_width};
	cufftPlanMany(&plan,2,n,inembed,1,inembed[0]*inembed[1],inembed,1,inembed[0]*inembed[1],
	                 CUFFT_Z2Z,1);
	cufftExecZ2Z(plan, d_fw, d_fw, iflag);

        cudaEventRecord(start);
        ier = cnufft_copygpumem_to_cpumem(nf1, nf2, M, fw_width, h_fw, d_fw, h_kx, d_kx,
                                          h_ky, d_ky, h_c, d_c);
	// Step 3: deconvolve and shuffle
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory DtoH\t %.3g ms\n", milliseconds);
#endif
	cufftDestroy(plan);
        cnufft_free_gpumemory(d_fw, d_kx, d_ky, d_c);
	return 0;
}
