#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include "spread.h"
#include "deconvolve.h"
#include "cufinufft.h"
#include "finufft/utils.h"

using namespace std;

int cnufft_copygpumem_to_cpumem_fk(int ms, int mt, CPX* h_fk, spread_devicemem *d_mem)
{
        checkCudaErrors(cudaMemcpy(h_fk,d_mem->fk,ms*mt*sizeof(CUCPX),cudaMemcpyDeviceToHost));
        return 0;
}

int cufinufft2d(int N1, int N2, int M, FLT* h_kx, FLT* h_ky, CPX* h_c, FLT tol, 
		int iflag, int nf1, int nf2, CPX* h_fk, spread_opts opts, 
		spread_devicemem* d_mem, FLT *fwkerhalf1, FLT *fwkerhalf2)
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
	ier = cnufft_allocgpumemory(N1, N2, nf1, nf2, M, &fw_width, opts, d_mem);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Allocate GPU memory\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	ier = cnufft_copycpumem_to_gpumem(M, h_kx, h_ky, h_c, nf1, nf2, fwkerhalf1, fwkerhalf2, d_mem);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory HtoD\t %.3g s\n", milliseconds/1000);
#endif

	// Step 1: Spread
	cudaEventRecord(start);
	ier = cnufftspread2d_gpu_subprob(nf1, nf2, fw_width, M, opts, d_mem);
	if(ier != 0 ){
		cout<<"error: cnufftspread2d_gpu_idriven"<<endl;
		return 0;
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Spread\t\t\t %.3g s\n", milliseconds/1000);
#endif
	// Step 2: Call FFT
	cudaEventRecord(start);
	cufftHandle plan;
	int ndata=1;
	int n[] = {nf2, nf1};
	int inembed[] = {nf2, fw_width};
	cufftPlanMany(&plan,2,n,inembed,1,inembed[0]*inembed[1],inembed,1,inembed[0]*inembed[1],
		      CUFFT_TYPE,ndata);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] CUFFT Plan\t\t %.3g s\n", milliseconds/1000);
#endif
	cudaEventRecord(start);
	CUFFT_EX(plan, d_mem->fw, d_mem->fw, iflag);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] CUFFT Exec\t\t %.3g s\n", milliseconds/1000);
#endif
	// Step 3: deconvolve and shuffle
	cudaEventRecord(start);
	cnufftdeconvolve2d_gpu(N1,N2,nf1,nf2,fw_width,opts,d_mem);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[time  ] Deconvolve\t\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	ier = cnufft_copygpumem_to_cpumem_fk(N1, N2, h_fk, d_mem);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory DtoH\t %.3g s\n", milliseconds/1000);
#endif
	cufftDestroy(plan);
	cnufft_free_gpumemory(opts, d_mem);
	return 0;
}
