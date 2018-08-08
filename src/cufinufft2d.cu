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
		int iflag, int nf1, int nf2, CPX* h_fw, spread_opts opts, spread_devicemem* d_mem)
{
	if(opts.pirange){
		for(int i=0; i<M; i++){
			h_kx[i]=RESCALE(h_kx[i], nf1, opts.pirange);
			h_ky[i]=RESCALE(h_ky[i], nf2, opts.pirange);
		}
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int ier;
	// Step 0: Allocate and transfer memory for GPU
#ifdef INFO
	cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
#endif

	int fw_width;
	cudaEventRecord(start);
	ier = cnufft_allocgpumemory(nf1, nf2, M, &fw_width, opts, d_mem);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Allocating GPU memory " << milliseconds <<" ms"<<endl;
#endif

	cudaEventRecord(start);
	ier = cnufft_copycpumem_to_gpumem(M, h_kx, h_ky, h_c, d_mem);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Copying memory from host to device " << milliseconds <<" s"<<endl;
#endif

	// Step 1: Spread
	cudaEventRecord(start);
	ier = cnufftspread2d_gpu_idriven(nf1, nf2, fw_width, M, opts, d_mem);
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
	int ndata=1;
	int n[] = {nf2, nf1};
	int inembed[] = {nf2, fw_width};
	cufftPlanMany(&plan,2,n,inembed,1,inembed[0]*inembed[1],inembed,1,inembed[0]*inembed[1],
			CUFFT_Z2Z,ndata);
	cufftExecZ2Z(plan, d_mem->fw, d_mem->fw, iflag);

	cudaEventRecord(start);
	ier = cnufft_copygpumem_to_cpumem(nf1, nf2, fw_width, h_fw, d_mem);
	// Step 3: deconvolve and shuffle
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory DtoH\t %.3g ms\n", milliseconds);
#endif
	cufftDestroy(plan);
	cnufft_free_gpumemory(opts, d_mem);
	return 0;
}
