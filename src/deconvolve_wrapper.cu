#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "cudeconvolve.h"

using namespace std;

/* Kernel for copying fw to fk with amplication by prefac/ker */
// Note: assume modeord=0: CMCL-compatible mode ordering in fk (from -N/2 up 
// to N/2-1)
__global__
void Deconvolve_2d(int ms, int mt, int nf1, int nf2, CUCPX* fw, CUCPX *fk, 
	FLT *fwkerhalf1, FLT *fwkerhalf2)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt; i+=blockDim.x*gridDim.x){
		int k1 = i % ms;
		int k2 = i / ms;
		int outidx = k1 + k2*ms;
		int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
		int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
		int inidx = w1 + w2*nf1;

		FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)];
		fk[outidx].x = fw[inidx].x/kervalue;
		fk[outidx].y = fw[inidx].y/kervalue;
	}
}

__global__
void Deconvolve_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, CUCPX* fw, 
	CUCPX *fk, FLT *fwkerhalf1, FLT *fwkerhalf2, FLT *fwkerhalf3)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt*mu; i+=blockDim.x*
		gridDim.x){
		int k1 = i % ms;
		int k2 = (i / ms) % mt;
		int k3 = (i / ms / mt);
		int outidx = k1 + k2*ms + k3*ms*mt;
		int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
		int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
		int w3 = k3-mu/2 >= 0 ? k3-mu/2 : nf3+k3-mu/2;
		int inidx = w1 + w2*nf1 + w3*nf1*nf2;

		FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)]*
			fwkerhalf3[abs(k3-mu/2)];
		fk[outidx].x = fw[inidx].x/kervalue;
		fk[outidx].y = fw[inidx].y/kervalue;
		//fk[outidx].x = kervalue;
		//fk[outidx].y = kervalue;
	}
}

/* Kernel for copying fk to fw with same amplication */
__global__
void Amplify_2d(int ms, int mt, int nf1, int nf2, CUCPX* fw, CUCPX *fk, 
	FLT *fwkerhalf1, FLT *fwkerhalf2)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt; i+=blockDim.x*gridDim.x){
		int k1 = i % ms;
		int k2 = i / ms;
		int inidx = k1 + k2*ms;
		int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
		int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
		int outidx = w1 + w2*nf1;

		FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)];
		fw[outidx].x = fk[inidx].x/kervalue;
		fw[outidx].y = fk[inidx].y/kervalue;
	}
}

__global__
void Amplify_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, CUCPX* fw, 
	CUCPX *fk, FLT *fwkerhalf1, FLT *fwkerhalf2, FLT *fwkerhalf3)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt*mu; 
		i+=blockDim.x*gridDim.x){
		int k1 = i % ms;
		int k2 = (i / ms) % mt;
		int k3 = (i / ms / mt);
		int inidx = k1 + k2*ms + k3*ms*mt;
		int w1 = k1-ms/2 >= 0 ? k1-ms/2 : nf1+k1-ms/2;
		int w2 = k2-mt/2 >= 0 ? k2-mt/2 : nf2+k2-mt/2;
		int w3 = k3-mu/2 >= 0 ? k3-mu/2 : nf3+k3-mu/2;
		int outidx = w1 + w2*nf1 + w3*nf1*nf2;

		FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)]*
			fwkerhalf3[abs(k3-mu/2)];
		fw[outidx].x = fk[inidx].x/kervalue;
		fw[outidx].y = fk[inidx].y/kervalue;
		//fw[outidx].x = fk[inidx].x;
		//fw[outidx].y = fk[inidx].y;
	}
}


int CUDECONVOLVE2D(CUFINUFFT_PLAN d_plan, int blksize)
/* 
	wrapper for deconvolution & amplication in 2D.

	Melody Shih 07/25/19
*/
{
	int ms=d_plan->ms;
	int mt=d_plan->mt;
	int nf1=d_plan->nf1;
	int nf2=d_plan->nf2;
	int nmodes=ms*mt;
	int maxbatchsize=d_plan->maxbatchsize;

	if(d_plan->spopts.spread_direction == 1){
		for(int t=0; t<blksize; t++){
			Deconvolve_2d<<<(nmodes+256-1)/256, 256>>>(ms, mt, nf1, nf2, 
				d_plan->fw+t*nf1*nf2,d_plan->fk+t*nmodes,d_plan->fwkerhalf1, 
				d_plan->fwkerhalf2);
		}
	}else{
		checkCudaErrors(cudaMemset(d_plan->fw,0,maxbatchsize*nf1*nf2*
			sizeof(CUCPX)));
		for(int t=0; t<blksize; t++){
			Amplify_2d<<<(nmodes+256-1)/256, 256>>>(ms, 
				mt, nf1, nf2, d_plan->fw+t*nf1*nf2, d_plan->fk+t*nmodes,
				d_plan->fwkerhalf1, d_plan->fwkerhalf2);
#ifdef DEBUG
			CPX* h_fw;
			h_fw = (CPX*) malloc(nf1*nf2*sizeof(CPX));
			checkCudaErrors(cudaMemcpy2D(h_fw,nf1*sizeof(CUCPX),d_plan->fw,
				nf1*sizeof(CUCPX),nf1*sizeof(CUCPX),nf2,
				cudaMemcpyDeviceToHost));
			for(int j=0; j<nf2; j++){
				for(int i=0; i<nf1; i++){
					printf("(%g,%g)",h_fw[i+j*nf1].real(),h_fw[i+j*nf1].imag());
				}
				printf("\n");
			}
			free(h_fw);
#endif
		}
	}
	return 0;
}

int CUDECONVOLVE3D(CUFINUFFT_PLAN d_plan, int blksize)
/* 
	wrapper for deconvolution & amplication in 3D.

	Melody Shih 07/25/19
*/
{
	int ms=d_plan->ms;
	int mt=d_plan->mt;
	int mu=d_plan->mu;
	int nf1=d_plan->nf1;
	int nf2=d_plan->nf2;
	int nf3=d_plan->nf3;
	int nmodes=ms*mt*mu;
	int maxbatchsize=d_plan->maxbatchsize;
	if(d_plan->spopts.spread_direction == 1){
		for(int t=0; t<blksize; t++){
			Deconvolve_3d<<<(nmodes+256-1)/256, 256>>>(ms, mt, mu, nf1, nf2, 
				nf3, d_plan->fw+t*nf1*nf2*nf3, d_plan->fk+t*nmodes, 
				d_plan->fwkerhalf1, d_plan->fwkerhalf2, d_plan->fwkerhalf3);
		}
	}else{
		checkCudaErrors(cudaMemset(d_plan->fw,0,maxbatchsize*nf1*nf2*nf3*
			sizeof(CUCPX)));
		for(int t=0; t<blksize; t++){
			Amplify_3d<<<(nmodes+256-1)/256, 256>>>(ms, mt, mu, nf1, nf2, nf3,
				d_plan->fw+t*nf1*nf2*nf3, d_plan->fk+t*nmodes, 
				d_plan->fwkerhalf1, d_plan->fwkerhalf2, d_plan->fwkerhalf3);
#ifdef DEBUG
			CPX* h_fw;
			h_fw = (CPX*) malloc(nf1*nf2*nf3*sizeof(CPX));
			checkCudaErrors(cudaMemcpy(h_fw,d_plan->fw,nf1*nf2*nf3*sizeof(CUCPX),
				cudaMemcpyDeviceToHost));
			for(int k=0; k<nf3; k++){
				for(int j=0; j<nf2; j++){
					for(int i=0; i<nf1; i++){
						printf("(%g,%g,%g)",h_fw[i+j*nf1+k*nf1*nf2].real(),
							h_fw[i+j*nf1+k*nf1*nf2].imag());
					}
					printf("\n");
				}
				printf("\n");
			}
			free(h_fw);
#endif
		}
	}
	return 0;
}
