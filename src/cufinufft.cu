#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include <cufinufft_eitherprec.h>
#include "cuspreadinterp.h"
#include "cudeconvolve.h"
#include "memtransfer.h"

using namespace std;

void SETUP_BINSIZE(int type, int dim, cufinufft_opts *opts)
{
	switch(dim)
	{
		case 2:
		{
			opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 32:
				opts->gpu_binsizex;
			opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 32:
				opts->gpu_binsizey;
			opts->gpu_binsizez = 1;
		}
		break;
		case 3:
		{
			switch(opts->gpu_method)
			{
				case 1:
				case 2:
				{
					opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 16:
						opts->gpu_binsizex;
					opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 16:
						opts->gpu_binsizey;
					opts->gpu_binsizez = (opts->gpu_binsizez < 0) ? 2:
						opts->gpu_binsizez;
				}
				break;
				case 4:
				{
					opts->gpu_obinsizex = (opts->gpu_obinsizex < 0) ? 8:
						opts->gpu_obinsizex;
					opts->gpu_obinsizey = (opts->gpu_obinsizey < 0) ? 8:
						opts->gpu_obinsizey;
					opts->gpu_obinsizez = (opts->gpu_obinsizez < 0) ? 8:
						opts->gpu_obinsizez;
					opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 4:
						opts->gpu_binsizex;
					opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 4:
						opts->gpu_binsizey;
					opts->gpu_binsizez = (opts->gpu_binsizez < 0) ? 4:
						opts->gpu_binsizez;
				}
				break;
			}
		}
		break;
	}
}

#ifdef __cplusplus
extern "C" {
#endif
int CUFINUFFT_MAKEPLAN(int type, int dim, int *nmodes, int iflag,
		       int ntransf, FLT tol, int maxbatchsize,
		       CUFINUFFT_PLAN *d_plan_ptr, cufinufft_opts *opts)
/*
	"plan" stage (in single or double precision).
        See ../docs/cppdoc.md for main user-facing documentation.
        Note that *d_plan_ptr in the args list was called simply *plan there.
        This is the remaining dev-facing doc:

This performs:
                (0) creating a new plan struct (d_plan), a pointer to which is passed
                    back by writing that pointer into *d_plan_ptr.
              	(1) set up the spread option, d_plan.spopts.
		(2) calculate the correction factor on cpu, copy the value from cpu to
		    gpu
		(3) allocate gpu arrays with size determined by number of fourier modes
		    and method related options that had been set in d_plan.opts
		(4) call cufftPlanMany and save the cufft plan inside cufinufft plan
        Variables and arrays inside the plan struct are set and allocated.

	Melody Shih 07/25/19. Use-facing moved to markdown, Barnett 2/16/21.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        if (opts == NULL) {
            // options might not be supplied to this function => assume device
            // 0 by default
            cudaSetDevice(0);
        } else {
            cudaSetDevice(opts->gpu_device_id);
        }

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int ier;

	/* allocate the plan structure, assign address to user pointer. */
	CUFINUFFT_PLAN d_plan = new CUFINUFFT_PLAN_S;
	*d_plan_ptr = d_plan;
        // Zero out your struct, (sets all pointers to NULL)
	memset(d_plan, 0, sizeof(*d_plan));


	/* If a user has not supplied their own options, assign defaults for them. */
	if (opts==NULL){    // use default opts
	  ier = CUFINUFFT_DEFAULT_OPTS(type, dim, &(d_plan->opts));
	  if (ier != 0){
	    printf("error: CUFINUFFT_DEFAULT_OPTS returned error %d.\n", ier);
	    return ier;
	  }
	} else {    // or read from what's passed in
	  d_plan->opts = *opts;    // keep a deep copy; changing *opts now has no effect
	}

	/* Setup Spreader */
	ier = setup_spreader_for_nufft(d_plan->spopts,tol,d_plan->opts);
	if (ier>1)                           // proceed if success or warning
	  return ier;

	d_plan->dim = dim;
	d_plan->ms = nmodes[0];
	d_plan->mt = nmodes[1];
	d_plan->mu = nmodes[2];

	SETUP_BINSIZE(type, dim, &d_plan->opts);
	BIGINT nf1=1, nf2=1, nf3=1;
	SET_NF_TYPE12(d_plan->ms, d_plan->opts, d_plan->spopts, &nf1,
				  d_plan->opts.gpu_obinsizex);
	if(dim > 1)
		SET_NF_TYPE12(d_plan->mt, d_plan->opts, d_plan->spopts, &nf2,
                      d_plan->opts.gpu_obinsizey);
	if(dim > 2)
		SET_NF_TYPE12(d_plan->mu, d_plan->opts, d_plan->spopts, &nf3,
                      d_plan->opts.gpu_obinsizez);
	int fftsign = (iflag>=0) ? 1 : -1;

	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->nf3 = nf3;
	d_plan->iflag = fftsign;
	d_plan->ntransf = ntransf;
	if (maxbatchsize==0)                    // implies: use a heuristic.
	   maxbatchsize = min(ntransf, 8);      // heuristic from test codes
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
			ier = ALLOCGPUMEM1D_PLAN(d_plan);
		}
		break;
		case 2:
		{
			ier = ALLOCGPUMEM2D_PLAN(d_plan);
		}
		break;
		case 3:
		{
			ier = ALLOCGPUMEM3D_PLAN(d_plan);
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

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_SETPTS(int M, FLT* d_kx, FLT* d_ky, FLT* d_kz, int N, FLT *d_s,
	FLT *d_t, FLT *d_u, CUFINUFFT_PLAN d_plan)
/*
	"setNUpts" stage (in single or double precision).

	In this stage, we
		(1) set the number and locations of nonuniform points
		(2) allocate gpu arrays with size determined by number of nupts
		(3) rescale x,y,z coordinates for spread/interp (on gpu, rescaled
		    coordinates are stored)
		(4) determine the spread/interp properties that only relates to the
		    locations of nupts (see 2d/spread2d_wrapper.cu,
		    3d/spread3d_wrapper.cu for what have been done in
		    function spread<dim>d_<method>_prop() )

        See ../docs/cppdoc.md for main user-facing documentation.
        Here is the old developer docs, which are useful only to translate
        the argument names from the user-facing ones:
        
	Input:
	M                 number of nonuniform points
	d_kx, d_ky, d_kz  gpu array of x,y,z locations of sources (each a size M
	                  FLT array) in [-pi, pi). set h_kz to "NULL" if dimension
	                  is less than 3. same for h_ky for dimension 1.
	N, d_s, d_t, d_u  not used for type1, type2. set to 0 and NULL.

	Input/Output:
	d_plan            pointer to a CUFINUFFT_PLAN_S. Variables and arrays inside
	                  the plan are set and allocated.

        Returned value:
        a status flag: 0 if success, otherwise an error occurred

Notes: the type FLT means either single or double, matching the
	precision of the library version called.

	Melody Shih 07/25/19; Barnett 2/16/21 moved out docs.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);


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
			ier = ALLOCGPUMEM1D_NUPTS(d_plan);
		}
		break;
		case 2:
		{
			ier = ALLOCGPUMEM2D_NUPTS(d_plan);
		}
		break;
		case 3:
		{
			ier = ALLOCGPUMEM3D_NUPTS(d_plan);
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
				ier = CUSPREAD2D_NUPTSDRIVEN_PROP(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_nupts_prop, method(%d)\n",
						d_plan->opts.gpu_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if(d_plan->opts.gpu_method==2){
				ier = CUSPREAD2D_SUBPROB_PROP(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_subprob_prop, method(%d)\n",
						d_plan->opts.gpu_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if(d_plan->opts.gpu_method==3){
				int ier = CUSPREAD2D_PAUL_PROP(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_paul_prop, method(%d)\n",
						d_plan->opts.gpu_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
		}
		break;
		case 3:
		{
			if(d_plan->opts.gpu_method==4){
				int ier = CUSPREAD3D_BLOCKGATHER_PROP(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_blockgather_prop, method(%d)\n",
						d_plan->opts.gpu_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
			if(d_plan->opts.gpu_method==1){
				ier = CUSPREAD3D_NUPTSDRIVEN_PROP(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n",
						d_plan->opts.gpu_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
			if(d_plan->opts.gpu_method==2){
				int ier = CUSPREAD3D_SUBPROB_PROP(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_subprob_prop, method(%d)\n",
						d_plan->opts.gpu_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
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

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}

int CUFINUFFT_EXECUTE(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN d_plan)
/*
	"exec" stage (single and double precision versions).

	The actual transformation is done here. Type and dimension of the
	transformation are defined in d_plan in previous stages.

        See ../docs/cppdoc.md for main user-facing documentation.

	Input/Output:
	d_c   a size d_plan->M CPX array on gpu (input for Type 1; output for Type
	      2)
	d_fk  a size d_plan->ms*d_plan->mt*d_plan->mu CPX array on gpu ((input for
	      Type 2; output for Type 1)

	Notes:
        i) Here CPX is a defined type meaning either complex<float> or complex<double>
	    to match the precision of the library called.
        ii) All operations are done on the GPU device (hence the d_* names)

	Melody Shih 07/25/19; Barnett 2/16/21.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

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
				ier = CUFINUFFT2D1_EXEC(d_c,  d_fk, d_plan);
			if(type == 2)
				ier = CUFINUFFT2D2_EXEC(d_c,  d_fk, d_plan);
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
			}
		}
		break;
		case 3:
		{
			if(type == 1)
				ier = CUFINUFFT3D1_EXEC(d_c,  d_fk, d_plan);
			if(type == 2)
				ier = CUFINUFFT3D2_EXEC(d_c,  d_fk, d_plan);
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
			}
		}
		break;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_DESTROY(CUFINUFFT_PLAN d_plan)
/*
	"destroy" stage (single and double precision versions).

	In this stage, we
		(1) free all the memories that have been allocated on gpu
		(2) delete the cuFFT plan

        Also see ../docs/cppdoc.md for main user-facing documentation.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// Can't destroy a Null pointer.
	if(!d_plan) {
                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);
		return 1;
        }

	if(d_plan->fftplan)
		cufftDestroy(d_plan->fftplan);

	switch(d_plan->dim)
	{
		case 1:
		{
			FREEGPUMEMORY1D(d_plan);
		}
		break;
		case 2:
		{
			FREEGPUMEMORY2D(d_plan);
		}
		break;
		case 3:
		{
			FREEGPUMEMORY3D(d_plan);
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

	/* free/destruct the plan */
	delete d_plan;
	/* set pointer to NULL now that we've hopefully free'd the memory. */
	d_plan = NULL;

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
	return 0;
}

int CUFINUFFT_DEFAULT_OPTS(int type, int dim, cufinufft_opts *opts)
/*
	Sets the default options in cufinufft_opts. This must be called
	before the user changes any options from default values.
	The resulting struct may then be passed (instead of NULL) to the last
	argument of cufinufft_plan().

	Options with prefix "gpu_" are used for gpu code.

	Notes:
	Values set in this function for different type and dimensions are preferable
	based on experiments. User can experiement with different settings by
	replacing them after calling this function.

	Melody Shih 07/25/19; Barnett 2/5/21.
*/
{
	int ier;
	opts->upsampfac = (FLT)2.0;

	/* following options are for gpu */
	opts->gpu_nstreams = 0;
	opts->gpu_kerevalmeth = 0; // using exp(sqrt())
	opts->gpu_sort = 1; // access nupts in an ordered way for nupts driven method

	opts->gpu_maxsubprobsize = 1024;
	opts->gpu_obinsizex = -1;
	opts->gpu_obinsizey = -1;
	opts->gpu_obinsizez = -1;

	opts->gpu_binsizex = -1;
	opts->gpu_binsizey = -1;
	opts->gpu_binsizez = -1;

	opts->gpu_spreadinterponly = 0; // default to do the whole nufft

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
			if(type == 1){
				opts->gpu_method = 2;
			}
			if(type == 2){
				opts->gpu_method = 1;
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
			if(type == 1){
				opts->gpu_method = 2;
			}
			if(type == 2){
				opts->gpu_method = 1;
			}
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
				return ier;
			}
		}
		break;
	}

        // By default, only use device 0
        opts->gpu_device_id = 0;

	return 0;
}
#ifdef __cplusplus
}
#endif
