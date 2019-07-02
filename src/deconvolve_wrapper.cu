#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "deconvolve.h"

using namespace std;
// Assume modeord=0: CMCL-compatible mode ordering in fk (from -N/2 up to N/2-1)
__global__
void Deconvolve_2d(int ms, int mt, int nf1, int nf2, int fw_width, CUCPX* fw, CUCPX *fk, 
                   FLT *fwkerhalf1, FLT *fwkerhalf2)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt; i+=blockDim.x*gridDim.x){
		int k1 = i % ms;
		int k2 = i / ms;
		int outidx = k1 + k2*ms;
		int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
		int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
		int inidx = w1 + w2*fw_width;

		FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)];
		fk[outidx].x = fw[inidx].x/kervalue;
		fk[outidx].y = fw[inidx].y/kervalue;
		//fk[outidx].x = kervalue;
		//fk[outidx].y = kervalue;
	}
}

__global__
void Amplify_2d(int ms, int mt, int nf1, int nf2, int fw_width, CUCPX* fw, CUCPX *fk, 
                   FLT *fwkerhalf1, FLT *fwkerhalf2)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt; i+=blockDim.x*gridDim.x){
		int k1 = i % ms;
		int k2 = i / ms;
		int inidx = k1 + k2*ms;
		int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
		int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
		int outidx = w1 + w2*fw_width;

		FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)];
		fw[outidx].x = fk[inidx].x/kervalue;
		fw[outidx].y = fk[inidx].y/kervalue;
		//fw[outidx].x = fk[inidx].x;
		//fw[outidx].y = fk[inidx].y;
	}
}

int cudeconvolve2d(const cufinufft_opts opts, cufinufft_plan *d_plan)
// ms = N1
// mt = N2
{
	int ms=d_plan->ms;
	int mt=d_plan->mt;
	int nf1=d_plan->nf1;
	int nf2=d_plan->nf2;
	int fw_width=d_plan->fw_width;
	int nmodes=ms*mt;
	int ntransfcufftplan=d_plan->ntransfcufftplan;
	if(opts.spread_direction == 1){
		for(int t=0; t<ntransfcufftplan; t++){
			Deconvolve_2d<<<(nmodes+256-1)/256, 256>>>(ms, mt, nf1, nf2, fw_width, 
							   	   d_plan->fw+t*nf1*nf2, 
								   d_plan->fk+t*nmodes,
							   	   d_plan->fwkerhalf1, 
								   d_plan->fwkerhalf2);
		}
	}else{
		checkCudaErrors(cudaMemset(d_plan->fw,0,nf1*nf2*sizeof(CUCPX)));
		Amplify_2d<<<(nmodes+256-1)/256, 256>>>(ms, mt, nf1, nf2, fw_width, 
							d_plan->fw, d_plan->fk,
						   	d_plan->fwkerhalf1, d_plan->fwkerhalf2);
#ifdef DEBUG
		CPX* h_fw;
		h_fw = (CPX*) malloc(nf1*nf2*sizeof(CPX));
		checkCudaErrors(cudaMemcpy2D(h_fw,nf1*sizeof(CUCPX),d_plan->fw,fw_width*sizeof(CUCPX),
                                    		 nf1*sizeof(CUCPX),nf2,cudaMemcpyDeviceToHost));
		for(int j=0; j<nf2; j++){
			for(int i=0; i<nf1; i++){
				printf("(%g,%g)",h_fw[i+j*nf1].real(),h_fw[i+j*nf1].imag());
			}
			printf("\n");
		}
		free(h_fw);
#endif
	}
	return 0;
}

