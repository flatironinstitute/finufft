#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include <cufinufft.h>
#include "cuspreadinterp.h"
#include "cudeconvolve.h"
#include "memtransfer.h"

using namespace std;

int cufinufft_makeplan(int type, int dim, int *nmodes, int iflag, 
	int ntransf, FLT tol, int maxbatchsize, cufinufft_plan *d_plan)
/*
	"plan" stage: 
	
	In this stage, we
		(1) set up the spread option, d_plan.spopts.
		(2) calculate the correction factor on cpu, copy the value from cpu to
		    gpu
		(3) allocate gpu arrays with size determined by number of fourier modes 
		    and method related options that had been set in d_plan.opts
		(4) call cufftPlanMany and save the cufft plan inside cufinufft plan
		
	Input:
	type    type of the transform, can be 1, 2, or 3 (3 not implemented yet))
	dim     dimension of the transform
	nmodes  a size 3 integer array, nmodes[d] is the number of modes in d 
	        dimension
	iflag   if >=0, uses + sign in exponential, otherwise - sign (int)
    ntransf number of transforms performed in exec stage
	tol     precision requested (>1e-16 for double precision, >1e-8 for single
	        precision)
	
	Input/Output:
	d_plan  a pointer to an instant of cufinuff_plan (definition in cufinufft.h) 
			    d_plan.opts is used for plan stage. Variables and arrays 
	        inside the plan are set and allocated
	
	Melody Shih 07/25/19
*/
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
	d_plan->maxbatchsize = maxbatchsize;
	d_plan->type = type;

	if(d_plan->type == 1)
		d_plan->spopts.spread_direction = 1; 
	if(d_plan->type == 2)
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
			
			//cufftCreate(&fftplan);
			//cufftPlan2d(&fftplan,n[0],n[1],CUFFT_TYPE);
			cufftPlanMany(&fftplan,dim,n,inembed,1,inembed[0]*inembed[1],
				inembed,1,inembed[0]*inembed[1],CUFFT_TYPE,maxbatchsize);
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
				CUFFT_TYPE,maxbatchsize);
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

int cufinufft_setNUpts(int M, FLT* d_kx, FLT* d_ky, FLT* d_kz, int N, FLT *d_s, 
	FLT *d_t, FLT *d_u, cufinufft_plan *d_plan)
/*
	"setNUpts" stage:
	
	In this stage, we
		(1) set the number and locations of nonuniform points
		(2) allocate gpu arrays with size determined by number of nupts
		(3) rescale x,y,z coordinates for spread/interp (on gpu, rescaled 
		    coordinates are stored)
		(4) determine the spread/interp properties that only relates to the 
		    locations of nupts (see 2d/spread2d_wrapper.cu, 
		    3d/spread3d_wrapper.cu for what have been done in 
		    function spread<dim>d_<method>_prop() )
	
	Input: 
	M                 number of nonuniform points
	d_kx, d_ky, d_kz  gpu array of x,y,z locations of sources (each a size M 
	                  FLT array) in [-pi, pi). set h_kz to "NULL" if dimension 
	                  is less than 3. same for h_ky for dimension 1.
	N, d_s, d_t, d_u  not used for type1, type2. set to 0 and NULL.

	Input/Output:
	d_plan            pointer to a cufinufft_plan. Variables and arrays inside 
	                  the plan are set and allocated.

	Melody Shih 07/25/19
*/
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int dim = d_plan->dim;

	d_plan->M = M;
#ifdef INFO
	printf("[info  ] 2d1: (ms,mt)=(%d,%d) (nf1, nf2, nf3)=(%d,%d,%d) nj=%d, ntransform = %d\n",
		d_plan->ms, d_plan->mt, d_plan->nf1, d_plan->nf2, nf3, d_plan->M,
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

	d_plan->kx = d_kx;
	if(dim > 1)
		d_plan->ky = d_ky;
	if(dim > 2)
		d_plan->kz = d_kz;
#if 0
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
#endif
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
			if(d_plan->opts.gpu_method==1){
				ier = cuspread2d_nuptsdriven_prop(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_nupts_prop, method(%d)\n", 
						d_plan->opts.gpu_method);
					return 1;
				}
			}
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
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tSetup Subprob properties %.3g s\n", 
		milliseconds/1000);
#endif

	return 0;
}

int cufinufft_exec(CUCPX* d_c, CUCPX* d_fk, cufinufft_plan *d_plan)
/*
	"exec" stage:
	
	The actual transformation is done in this stage. Type and dimension of the 
	transformantion are defined in d_plan in previous stages. 

	Input/Output:
	d_c   a size d_plan->M CPX array on gpu (input for Type 1; output for Type 
	      2)
	d_fk  a size d_plan->ms*d_plan->mt*d_plan->mu CPX array on gpu ((input for 
	      Type 2; output for Type 1)

	Notes:
	For now, we assume both h_c, h_fk arrays are on cpu so this stage includes
    copying the arrays from/to cpu to/from gpu.

	Melody Shih 07/25/19
*/
{
	int ier;
	int type=d_plan->type;
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
			if(type == 1)
				ier = cufinufft2d1_exec(d_c,  d_fk, d_plan);
			if(type == 2)
				ier = cufinufft2d2_exec(d_c,  d_fk, d_plan);
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
			}
		}
		break;
		case 3:
		{
			if(type == 1)
				ier = cufinufft3d1_exec(d_c,  d_fk, d_plan);
			if(type == 2)
				ier = cufinufft3d2_exec(d_c,  d_fk, d_plan);
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
			}
		}
		break;
	}
	return ier;
}

int cufinufft_destroy(cufinufft_plan *d_plan)
/*
	"destroy" stage:

	In this stage, we
		(1) free all the memories that have been allocated on gpu
		(2) delete the cuFFT plan

*/
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
			freegpumemory1d(d_plan);
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

int cufinufft_default_opts(int type, int dim, cufinufft_opts &opts)
/*
	"default_opts" stage:
	
	In this stage, the default options in cufinufft_opts are set. 
  Options with prefix "gpu_" are used for gpu code. 

	Notes:
	Values set in this function for different type and dimensions are preferable 
	based on experiments. User can experiement with different settings by 
	replacing them after calling this function.

	Melody Shih 07/25/19
*/
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
			opts.gpu_maxsubprobsize = 1024;
			if(type == 1){
				opts.gpu_method = 2;
				opts.gpu_binsizex = 32;
				opts.gpu_binsizey = 32;
				opts.gpu_binsizez = 1;
			}
			if(type == 2){
				opts.gpu_method = 1;
				opts.gpu_binsizex = 32;
				opts.gpu_binsizey = 32;
				opts.gpu_binsizez = 1;
			}
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
				return ier;
			}
		}
		break;
		case 3:
		{
			opts.gpu_maxsubprobsize = 1024;
			if(type == 1){
				opts.gpu_method = 2;
				opts.gpu_binsizex = 16;
				opts.gpu_binsizey = 16;
				opts.gpu_binsizez = 2;
			}
			if(type == 2){
				opts.gpu_method = 1;
				opts.gpu_binsizex = 16;
				opts.gpu_binsizey = 16;
				opts.gpu_binsizez = 2;
			}
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
				return ier;
			}
		}
		break;
	}

	opts.upsampfac = (FLT)2.0;
	return 0;
}
