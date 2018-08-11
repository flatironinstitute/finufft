#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "spread.h"
#include "memtransfer.h"

using namespace std;

// This function includes device memory allocation, transfer, free
int cufinufft_interp2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX *h_c, spread_opts opts, cufinufft_plan* d_plan)
{
	if(opts.spread_direction!=2){
		printf("spread direction not set\n");
		return 1;
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	
	d_plan->ms = ms;
        d_plan->mt = mt;
        d_plan->nf1 = nf1;
        d_plan->nf2 = nf2;
	d_plan->M = M;

        d_plan->h_kx = h_kx;
        d_plan->h_ky = h_ky;
        d_plan->h_c = h_c;
	d_plan->h_fw = h_fw;
	d_plan->h_fwkerhalf1 = NULL;
	d_plan->h_fwkerhalf2 = NULL;

	if(opts.pirange){
		for(int i=0; i<M; i++){
			h_kx[i]=RESCALE(h_kx[i], nf1, opts.pirange);
			h_ky[i]=RESCALE(h_ky[i], nf2, opts.pirange);
		}
	}
	cudaEventRecord(start);
	ier = allocgpumemory(opts, d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Allocate GPU memory\t %.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	ier = copycpumem_to_gpumem(opts, d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory HtoD\t %.3g ms\n", milliseconds);
#endif
	ier = cuinterp2d(opts, d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Spread\t\t\t %.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	ier = copygpumem_to_cpumem_c(d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory DtoH\t %.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	free_gpumemory(opts, d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	return ier;
}

// a wrapper of different methods of spreader
int cuinterp2d( spread_opts opts, cufinufft_plan* d_plan)
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int fw_width = d_plan->fw_width;
	int M = d_plan->M;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	switch(opts.method)
	{
		case 1:
			{
				cudaEventRecord(start);
				ier = cuinterp2d_idriven(nf1, nf2, fw_width, M, opts, d_plan);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_idriven"<<endl;
					return 1;
				}
			}
			break;
		case 5:
			{
				cudaEventRecord(start);
				ier = cuinterp2d_subprob(nf1, nf2, fw_width, M, opts, d_plan);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_hybrid"<<endl;
					return 1;
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1 or 5"<<endl;
			return 2;
	}
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Interp " << milliseconds <<" ms"<<endl;
#endif
	return ier;
}

int cuinterp2d_idriven(int nf1, int nf2, int fw_width, int M, spread_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	cudaEventRecord(start);
	Interp_2d_Idriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c, d_fw, M, ns,
						       nf1, nf2, es_c, es_beta, fw_width);

#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_2d_Idriven \t%.3g ms\n", milliseconds);
#endif
	return 0;
}

int cuinterp2d_subprob(int nf1, int nf2, int fw_width, int M, spread_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;
	int maxsubprobsize=opts.maxsubprobsize;

	// assume that bin_size_x > ns/2;
	int bin_size_x=opts.bin_size_x;
	int bin_size_y=opts.bin_size_y;
	int numbins[2];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<opts.bin_size_x<<"x"<<opts.bin_size_y<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_sortidx = d_plan->sortidx;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;
	d_plan->subprob_to_bin = NULL;
	int *d_subprob_to_bin = d_plan->subprob_to_bin;
	d_plan->temp_storage = NULL;
	void *d_temp_storage = d_plan->temp_storage;

	cudaEventRecord(start);
	checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*sizeof(int)));
	CalcBinSize_noghost_2d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,bin_size_x,bin_size_y,
			numbins[0],numbins[1],d_binsize,
			d_kx,d_ky,d_sortidx);
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcBinSize_noghost_2d \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int *h_binsize;// For debug
	h_binsize     = (int*)malloc(numbins[0]*numbins[1]*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*sizeof(int),
				cudaMemcpyDeviceToHost));
	cout<<"[debug ] bin size:"<<endl;
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<" bin["<<setw(3)<<i<<","<<setw(3)<<j<<"]="<<h_binsize[i+j*numbins[0]];
		}
		cout<<endl;
	}
	free(h_binsize);
	cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

	cudaEventRecord(start);
	int n=numbins[0]*numbins[1];
	size_t temp_storage_bytes = 0;
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_binsize, d_binstartpts, n));
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes)); // Allocate temporary storage for inclusive prefix scan
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_binsize, d_binstartpts, n));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel BinStartPts_2d \t\t\t%.3g ms\n", milliseconds);
#endif

#ifdef DEBUG
	int *h_binstartpts;
	h_binstartpts = (int*)malloc((numbins[0]*numbins[1])*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]*numbins[1])*sizeof(int),
				cudaMemcpyDeviceToHost));
	cout<<"[debug ] Result of scan bin_size array:"<<endl;
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<"bin["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)<<h_binstartpts[i+j*numbins[0]];
		}
		cout<<endl;
	}
	free(h_binstartpts);
	cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

	cudaEventRecord(start);
	CalcInvertofGlobalSortIdx_2d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,bin_size_y,numbins[0],
			numbins[1],d_binstartpts,d_sortidx,
			d_kx,d_ky,d_idxnupts);
#ifdef DEBUG
	int *h_idxnupts;
	h_idxnupts = (int*)malloc(M*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_idxnupts,d_idxnupts,M*sizeof(int),cudaMemcpyDeviceToHost));
	for (int i=0; i<M; i++){
		cout <<"[debug ] idx="<< h_idxnupts[i]<<endl;
	}
	free(h_idxnupts);
#endif
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcInvertofGlobalSortIdx_2d \t%.3g ms\n", milliseconds);
#endif

	/* --------------------------------------------- */
	//        Determining Subproblem properties      //
	/* --------------------------------------------- */
	cudaEventRecord(start);
	CalcSubProb_2d<<<(M+1024-1)/1024, 1024>>>(d_binsize,d_numsubprob,maxsubprobsize,numbins[0]*numbins[1]);
#ifdef DEBUG
	int* h_numsubprob;
	h_numsubprob = (int*) malloc(n*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_numsubprob,d_numsubprob,numbins[0]*numbins[1]*sizeof(int),
				cudaMemcpyDeviceToHost));
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<"nsub["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)<<h_numsubprob[i+j*numbins[0]];
		}
		cout<<endl;
	}
	free(h_numsubprob);
#endif
	// Scanning the same length array, so we don't need calculate temp_storage_bytes here
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_numsubprob, d_subprobstartpts+1, n));
	checkCudaErrors(cudaMemset(d_subprobstartpts,0,sizeof(int)));

#ifdef DEBUG
	printf("[debug ] Subproblem start points\n");
	int* h_subprobstartpts;
	h_subprobstartpts = (int*) malloc((n+1)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_subprobstartpts,d_subprobstartpts,(n+1)*sizeof(int),
				cudaMemcpyDeviceToHost));
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<"nsub["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)<<h_subprobstartpts[i+j*numbins[0]];
		}
		cout<<endl;
	}
	printf("[debug ] Total number of subproblems = %d\n", h_subprobstartpts[n]);
	free(h_subprobstartpts);
#endif

	int totalnumsubprob;
	checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[n],sizeof(int),
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));
	MapBintoSubProb_2d<<<(numbins[0]*numbins[1]+1024-1)/1024, 1024>>>(d_subprob_to_bin, 
			d_subprobstartpts,
			d_numsubprob,
			numbins[0]*numbins[1]);
#ifdef DEBUG
	printf("[debug ] Map Subproblem to Bins\n");
	int* h_subprob_to_bin;
	h_subprob_to_bin = (int*) malloc((totalnumsubprob)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_subprob_to_bin,d_subprob_to_bin,(totalnumsubprob)*sizeof(int),
				cudaMemcpyDeviceToHost));
	for(int j=0; j<totalnumsubprob; j++){
		cout<<"[debug ] ";
		cout <<"nsub["<<j<<"] = "<<setw(2)<<h_subprob_to_bin[j];
		cout<<endl;
	}
	free(h_subprob_to_bin);
#endif
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Subproblem to Bin map\t\t%.3g ms\n", milliseconds);
#endif
	FLT sigma=opts.upsampfac;
	cudaEventRecord(start);
	size_t sharedplanorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*sizeof(CUCPX);
	if(sharedplanorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	Interp_2d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(d_kx, d_ky, d_c,
			d_fw, M, ns, nf1, nf2,
			es_c, es_beta, sigma, fw_width,
			d_binstartpts, d_binsize,
			bin_size_x, bin_size_y,
			d_subprob_to_bin, d_subprobstartpts,
			d_numsubprob, maxsubprobsize,
			numbins[0], numbins[1], d_idxnupts);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_2d_Subprob_V2 \t\t%.3g ms\n", milliseconds);
#endif
	return 0;
}
