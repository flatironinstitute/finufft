#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include "spreadinterp.h"
#include "memtransfer.h"
#include "deconvolve.h"
#include "cufinufft.h"
#include "../finufft/utils.h"
#include "../finufft/common.h"

using namespace std;

int cufinufft_makeplan(finufft_type type, int dim, int *nmodes, int iflag, 
	int ntransf, FLT tol, int ntransfcufftplan, cufinufft_plan *d_plan)
{

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier = setup_spreader_for_nufft(d_plan->spopts,tol,d_plan->opts);

	d_plan->dim = dim;
	d_plan->ms = nmodes[0];
	d_plan->mt = nmodes[1];
	d_plan->mu = nmodes[2];

	int nf1=1, nf2=1, nf3=1;
	set_nf_type12(d_plan->ms, d_plan->opts, d_plan->spopts, &nf1);
	if(dim > 1)
		set_nf_type12(d_plan->mt, d_plan->opts, d_plan->spopts, &nf2); 
	if(dim > 2)
		set_nf_type12(d_plan->mu, d_plan->opts, d_plan->spopts, &nf3);
	int fftsign = (iflag>=0) ? 1 : -1;

	d_plan->nf1 = nf1;	
	d_plan->nf2 = nf2;	
	d_plan->nf3 = nf3;	
	d_plan->iflag = fftsign;
	d_plan->ntransf = ntransf;
	d_plan->ntransfcufftplan = ntransfcufftplan;
	d_plan->type = type;

	if(d_plan->type == type1)
		d_plan->spopts.spread_direction = 1; 
	if(d_plan->type == type2)
		d_plan->spopts.spread_direction = 2; 
	// this may move to gpu
	CNTime timer; timer.start();
	FLT *fwkerhalf1, *fwkerhalf2, *fwkerhalf3;

	fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
	onedim_fseries_kernel(nf1, fwkerhalf1, d_plan->spopts);
	if(dim > 1){	
		fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
		onedim_fseries_kernel(nf2, fwkerhalf2, d_plan->spopts);
	}
	if(dim > 2){	
		fwkerhalf3 = (FLT*)malloc(sizeof(FLT)*(nf3/2+1));
		onedim_fseries_kernel(nf3, fwkerhalf3, d_plan->spopts);
	}
#ifdef TIME
	printf("[time  ] \tkernel fser (ns=%d):\t %.3g s\n", d_plan->spopts.nspread,
		timer.elapsedsec());
#endif

	cudaEventRecord(start);
	switch(d_plan->dim)
	{
		case 1:
		{
			ier = allocgpumem1d_plan(d_plan);
		}
		break;
		case 2:
		{
			ier = allocgpumem2d_plan(d_plan);
		}
		break;
		case 3:
		{
			ier = allocgpumem3d_plan(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory plan %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf1,fwkerhalf1,(nf1/2+1)*
		sizeof(FLT),cudaMemcpyHostToDevice));
	if(dim > 1)
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf2,fwkerhalf2,(nf2/2+1)*
			sizeof(FLT),cudaMemcpyHostToDevice));
	if(dim > 2)
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf3,fwkerhalf3,(nf3/2+1)*
			sizeof(FLT),cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy fwkerhalf1,2 HtoD\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	cufftHandle fftplan;
	switch(d_plan->dim)
	{
		case 1:
		{
			cerr<<"Not implemented yet"<<endl;
		}
		break;
		case 2:
		{
			int n[] = {nf2, nf1};
			int inembed[] = {nf2, nf1};
			cufftPlanMany(&fftplan,dim,n,inembed,1,inembed[0]*inembed[1],
				inembed,1,inembed[0]*inembed[1],CUFFT_TYPE,ntransfcufftplan);
		}
		break;
		case 3:
		{
			int dim = 3;
			int n[] = {nf3, nf2, nf1};
			int inembed[] = {nf3, nf2, nf1};
			int istride = 1;
			cufftPlanMany(&fftplan,dim,n,inembed,istride,inembed[0]*inembed[1]*
				inembed[2],inembed,istride,inembed[0]*inembed[1]*inembed[2],
				CUFFT_TYPE,ntransfcufftplan);
		}
		break;
	}
	d_plan->fftplan = fftplan;
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCUFFT Plan\t\t %.3g s\n", milliseconds/1000);
#endif
	free(fwkerhalf1);
	if(dim > 1)
		free(fwkerhalf2);
	if(dim > 2)
		free(fwkerhalf3);
	return ier;
}

int cufinufft_setNUpts(int M, FLT* h_kx, FLT* h_ky, FLT* h_kz, int N, FLT *h_s, 
	FLT *h_t, FLT *h_u, cufinufft_plan *d_plan)
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int dim = d_plan->dim;

	d_plan->M = M;
#ifdef INFO
	printf("[info  ] 2d1: (ms,mt)=(%d,%d) (nf1, nf2)=(%d,%d) nj=%d, ntransform = %d\n",
		d_plan->ms, d_plan->mt, d_plan->nf1, d_plan->nf2, d_plan->M,
		d_plan->ntransf);
#endif
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	cudaEventRecord(start);
	switch(d_plan->dim)
	{
		case 1:
		{
			ier = allocgpumem1d_nupts(d_plan);
		}
		break;
		case 2:
		{
			ier = allocgpumem2d_nupts(d_plan);
		}
		break;
		case 3:
		{
			ier = allocgpumem3d_nupts(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory NUpts%.3g s\n", milliseconds/1000);
#endif

	// Copy memory to device
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->kx,h_kx,d_plan->M*sizeof(FLT),
		cudaMemcpyHostToDevice));
	if(dim > 1)
		checkCudaErrors(cudaMemcpy(d_plan->ky,h_ky,d_plan->M*sizeof(FLT),
			cudaMemcpyHostToDevice));
	if(dim > 2)
		checkCudaErrors(cudaMemcpy(d_plan->kz,h_kz,d_plan->M*sizeof(FLT),
			cudaMemcpyHostToDevice));
		
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy kx,ky,kz HtoD\t %.3g s\n", milliseconds/1000);
#endif

	if(d_plan->spopts.pirange == 1){
		cudaEventRecord(start);
		switch(d_plan->dim)
		{
			case 1:
			{
				cerr<<"Not implemented yet"<<endl;
			}
			break;
			case 2:
			{
				RescaleXY_2d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,d_plan->kx, 
					d_plan->ky);
				d_plan->spopts.pirange = 0;
			}
			break;
			case 3:
			{
				RescaleXY_3d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,nf3,d_plan->kx,
					d_plan->ky,d_plan->kz);
				d_plan->spopts.pirange = 0;
			}
			break;
		}	
#ifdef SPREADTIME
		float milliseconds;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ]\tRescaleXYZ\t\t %.3g ms\n", milliseconds);
#endif
	}

	switch(d_plan->dim)
	{
		case 1:
		{
			cerr<<"Not implemented yet"<<endl;
		}
		break;
		case 2:
		{
			if(d_plan->opts.gpu_method==2){
				ier = cuspread2d_subprob_prop(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_subprob_prop, method(%d)\n", 
						d_plan->opts.gpu_method);
					return 1;
				}
			}
			if(d_plan->opts.gpu_method==3){
				int ier = cuspread2d_paul_prop(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_paul_prop, method(%d)\n", 
						d_plan->opts.gpu_method);
					return 1;
				}
			}
		}
		break;
		case 3:
		{
			if(d_plan->opts.gpu_method==4){
				int ier = cuspread3d_blockgather_prop(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_blockgather_prop, method(%d)\n", 
						d_plan->opts.gpu_method);
					return 0;
				}
			}
			if(d_plan->opts.gpu_method==1){
				ier = cuspread3d_nuptsdriven_prop(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n", 
						d_plan->opts.gpu_method);
					return 0;
				}
			}
			if(d_plan->opts.gpu_method==2){
				int ier = cuspread3d_subprob_prop(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_subprob_prop, method(%d)\n", 
						d_plan->opts.gpu_method);
					return 0;
				}
			}
		}
		break;
	}	

	return 0;
}

int cufinufft_exec(CPX* h_c, CPX* h_fk, cufinufft_plan *d_plan)
{
	int ier;
	finufft_type type=d_plan->type;
	switch(d_plan->dim)
	{
		case 1:
		{
			cerr<<"Not Implemented yet"<<endl;
			ier = 1;
		}
		break;
		case 2:
		{
			if(type == type1)
				ier = cufinufft2d1_exec(h_c,  h_fk, d_plan);
			if(type == type2)
				ier = cufinufft2d2_exec(h_c,  h_fk, d_plan);
			if(type == type3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
			}
		}
		break;
		case 3:
		{
			if(type == type1)
				ier = cufinufft3d1_exec(h_c,  h_fk, d_plan);
			if(type == type2)
				ier = cufinufft3d2_exec(h_c,  h_fk, d_plan);
			if(type == type3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
			}
		}
		break;
	}
	return ier;
}

int cufinufft_destroy(cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cufftDestroy(d_plan->fftplan);
	switch(d_plan->dim)
	{
		case 1:
		{
			freegpumemory2d(d_plan);
		}
		break;
		case 2:
		{
			freegpumemory2d(d_plan);
		}
		break;
		case 3:
		{
			freegpumemory3d(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tFree gpu memory\t\t %.3g s\n", milliseconds/1000);
#endif
	return 0;
}

int cufinufft_default_opts(finufft_type type, int dim, nufft_opts &opts)
{
	int ier;
	/* following options are for gpu */
	opts.gpu_nstreams = 0;
	opts.gpu_kerevalmeth = 1; // using Horner ppval
	opts.gpu_sort = 1; // access nupts in an ordered way for nupts driven method

	opts.gpu_maxsubprobsize = 1024;
	opts.gpu_obinsizex = 8;
	opts.gpu_obinsizey = 8;
	opts.gpu_obinsizez = 8;

	opts.gpu_binsizex = 8;
	opts.gpu_binsizey = 8;
	opts.gpu_binsizez = 2;

	switch(dim)
	{
		case 1:
		{
			cerr<<"Not Implemented yet"<<endl;
			ier = 1;
			return ier;
		}
		case 2:
		{
			if(type == type1){
				opts.gpu_method = 2;
				opts.gpu_binsizex = 32;
				opts.gpu_binsizey = 32;
				opts.gpu_binsizez = 1;
			}
			if(type == type2){
				opts.gpu_method = 1;
			}
			if(type == type3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
				return ier;
			}
		}
		break;
		case 3:
		{
			if(type == type1){
				opts.gpu_method = 2;
				opts.gpu_binsizex = 16;
				opts.gpu_binsizey = 16;
				opts.gpu_binsizez = 2;
			}
			if(type == type2){
				opts.gpu_method = 1;
				opts.gpu_binsizex = 16;
				opts.gpu_binsizey = 16;
				opts.gpu_binsizez = 2;
			}
			if(type == type3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
				return ier;
			}
		}
		break;
	}

	opts.upsampfac = (FLT)2.0;   // sigma: either 2.0, or 1.25 for smaller RAM, FFTs

	/* following options are not used in gpu code */
	opts.chkbnds = -1;
	opts.debug = -1;
	opts.spread_debug = -1;
	opts.spread_sort = -1;        // use heuristic rule for whether to sort
	opts.spread_kerevalmeth = -1; // 0: direct exp(sqrt()), 1: Horner ppval
	opts.spread_kerpad = -1;      // (relevant iff kerevalmeth=0)

	return 0;
}
