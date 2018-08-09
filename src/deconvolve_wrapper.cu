#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "deconvolve.h"

using namespace std;

__global__
void Deconvolve_2d(int ms, int mt, int nf1, int nf2, int fw_width, CUCPX* fw, CUCPX *fk, 
                   FLT *fwkerhalf1, FLT *fwkerhalf2)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<ms*mt; i+=blockDim.x*gridDim.x){
		int k1 = i % ms;
		int k2 = i / ms;
		int outidx = k1 + k2*ms;
		int w1 = k1-ms/2 > 0 ? k1-ms/2 : nf1+k1-ms/2;
		int w2 = k2-mt/2 > 0 ? k2-mt/2 : nf2+k2-mt/2;
		int inidx = w1 + w2*fw_width;

		FLT kervalue = fwkerhalf1[abs(k1-ms/2)]*fwkerhalf2[abs(k2-mt/2)];
		fk[outidx].x = fw[inidx].x/kervalue;
		fk[outidx].y = fw[inidx].y/kervalue;
		//fk[outidx].x = kervalue;
		//fk[outidx].y = kervalue;
	}
}

int cnufftdeconvolve2d_gpu(int ms, int mt, int nf1, int nf2, int fw_width, spread_opts opts, spread_devicemem *d_mem)
// ms = N1
// mt = N2
{
	int nmodes=ms*mt;
	Deconvolve_2d<<<(nmodes+256-1)/256, 256>>>(ms, mt, nf1, nf2, fw_width, d_mem->fw, d_mem->fk,
						   d_mem->fwkerhalf1, d_mem->fwkerhalf2);
	return 0;
}

