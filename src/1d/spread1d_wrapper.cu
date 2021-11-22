#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuComplex.h>
#include "../cuspreadinterp.h"
#include "../memtransfer.h"

using namespace std;

int CUFINUFFT_SPREAD1D(int nf1, CUCPX* d_fw, int M, FLT *d_kx, CUCPX *d_c, 
	CUFINUFFT_PLAN d_plan)
/*
	This c function is written for only doing 1D spreading. See
	test/spread1d_test.cu for usage.

	note: not allocate,transfer and free memories on gpu.
	Melody Shih 11/21/21
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_plan->kx = d_kx;
	d_plan->c  = d_c;
	d_plan->fw = d_fw;

	int ier;
	d_plan->nf1 = nf1;
	d_plan->M = M;
	d_plan->maxbatchsize = 1;

	cudaEventRecord(start);
	ier = ALLOCGPUMEM1D_PLAN(d_plan);
	ier = ALLOCGPUMEM1D_NUPTS(d_plan);

	if(d_plan->opts.gpu_method == 1){
		ier = CUSPREAD1D_NUPTSDRIVEN_PROP(nf1,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread1d_nuptsdriven_prop, method(%d)\n",
				d_plan->opts.gpu_method);
			return ier;
		}
	}

	if(d_plan->opts.gpu_method == 2){
		ier = CUSPREAD1D_SUBPROB_PROP(nf1,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread1d_subprob_prop, method(%d)\n",
				d_plan->opts.gpu_method);
			return ier;
		}
	}

#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Obtain Spread Prop\t %.3g ms\n", milliseconds);
#endif

	cudaEventRecord(start);
	ier = CUSPREAD1D(d_plan,1);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Spread (%d)\t\t %5.3f ms\n", d_plan->opts.gpu_method,
		milliseconds);
#endif

	cudaEventRecord(start);
	FREEGPUMEMORY1D(d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	return ier;
}

int CUSPREAD1D(CUFINUFFT_PLAN d_plan, int blksize)
/*
	A wrapper for different spreading methods.

	Methods available:
	(1) Non-uniform points driven
	(2) Subproblem

	Melody Shih 11/21/21
*/
{
	int nf1 = d_plan->nf1;
	int M = d_plan->M;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				cudaEventRecord(start);
				ier = CUSPREAD1D_NUPTSDRIVEN(nf1, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cnufftspread1d_gpu_nuptsdriven"<<endl;
					return 1;
				}
			}
			break;
		case 2:
			{
				cudaEventRecord(start);
				ier = CUSPREAD1D_SUBPROB(nf1, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cnufftspread1d_gpu_subprob"<<endl;
					return 1;
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1,2"<<endl;
			return 2;
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Spread " << milliseconds <<" ms"<<endl;
#endif
	return ier;
}

int CUSPREAD1D_NUPTSDRIVEN_PROP(int nf1, int M, CUFINUFFT_PLAN d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if(d_plan->opts.gpu_sort){

		int bin_size_x=d_plan->opts.gpu_binsizex;
		if(bin_size_x < 0){
			cout<<"error: invalid binsize (binsizex) = ("<<bin_size_x<<")"<<endl;
			return 1; 
		}

		int numbins = ceil((FLT) nf1/bin_size_x);
#ifdef DEBUG
		cout<<"[debug ] Dividing the uniform grids to bin size["
			<<d_plan->opts.gpu_binsizex<<"]"<<endl;
		cout<<"[debug ] numbins = ["<<numbins<<"]"<<endl;
#endif

		FLT*   d_kx = d_plan->kx;
#ifdef DEBUG
		FLT *h_kx;
		h_kx = (FLT*)malloc(M*sizeof(FLT));

		checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),
			cudaMemcpyDeviceToHost));
		for(int i=0; i<M; i++){
			cout<<"[debug ] ";
			cout <<"("<<setw(3)<<h_kx[i]<<")"<<endl;
		}
#endif
		int *d_binsize = d_plan->binsize;
		int *d_binstartpts = d_plan->binstartpts;
		int *d_sortidx = d_plan->sortidx;
		int *d_idxnupts = d_plan->idxnupts;

		int pirange = d_plan->spopts.pirange;

		cudaEventRecord(start);
		checkCudaErrors(cudaMemset(d_binsize,0,numbins*sizeof(int)));
		CalcBinSize_noghost_1d<<<(M+1024-1)/1024, 1024>>>(M,nf1,
			bin_size_x,numbins,d_binsize,d_kx,d_sortidx,pirange);
#ifdef SPREADTIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel CalcBinSize_noghost_1d \t\t%.3g ms\n",
			milliseconds);
#endif
#ifdef DEBUG
		int *h_binsize;// For debug
		h_binsize     = (int*)malloc(numbins*sizeof(int));
		checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins*sizeof(int),
			cudaMemcpyDeviceToHost));
		cout<<"[debug ] bin size:"<<endl;
		cout<<"[debug ] ";
		for(int i=0; i<numbins; i++){
			if(i!=0) cout<<" ";
			cout <<"bin["<<setw(1)<<i<<"]="<<h_binsize[i];
		}
		cout<<endl;
		free(h_binsize);
		cout<<"[debug ] ------------------------------------------------"<<endl;

		int *h_sortidx;
		h_sortidx = (int*)malloc(M*sizeof(int));

		checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
			cudaMemcpyDeviceToHost));

		for(int i=0; i<M; i++){
			if(h_sortidx[i] < 0){
				cout<<"[debug ] ";
				cout <<"point["<<setw(3)<<i<<"]="<<setw(3)<<h_sortidx[i]<<endl;
				cout<<"[debug ] ";
				printf("(%10.10f) ", RESCALE(h_kx[i],nf1,pirange));
				printf("(%10.10f) ", RESCALE(h_kx[i],nf1,pirange)/32);
				printf("(%f)\n", floor(RESCALE(h_kx[i],nf1,pirange)/32));
			}
		}
#endif
		cudaEventRecord(start);
		int n=numbins;
		thrust::device_ptr<int> d_ptr(d_binsize);
		thrust::device_ptr<int> d_result(d_binstartpts);
		thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
#ifdef SPREADTIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel BinStartPts_1d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
		int *h_binstartpts;
		h_binstartpts = (int*)malloc((numbins)*sizeof(int));
		checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins)
			*sizeof(int),cudaMemcpyDeviceToHost));
		cout<<"[debug ] Result of scan bin_size array:"<<endl;
		cout<<"[debug ] ";
		for(int i=0; i<numbins; i++){
			if(i!=0) cout<<" ";
			cout <<"bin["<<setw(1)<<i<<"]="<<h_binstartpts[i];
		}
		cout<<endl;
		free(h_binstartpts);
		cout<<"[debug ] ------------------------------------------------"<<endl;
#endif
		cudaEventRecord(start);
		CalcInvertofGlobalSortIdx_1d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
			numbins,d_binstartpts,d_sortidx,d_kx,d_idxnupts,pirange,nf1);
#ifdef SPREADTIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel CalcInvertofGlobalSortIdx_1d \t%.3g ms\n",
			milliseconds);
#endif
#ifdef DEBUG
		int *h_idxnupts;
		h_idxnupts = (int*)malloc(M*sizeof(int));
		checkCudaErrors(cudaMemcpy(h_idxnupts,d_idxnupts,M*sizeof(int),
					cudaMemcpyDeviceToHost));
		for (int i=0; i<M; i++){
			cout <<"[debug ] idx="<< h_idxnupts[i]<<endl;
		}
		free(h_idxnupts);
#endif
	}else{
		int *d_idxnupts = d_plan->idxnupts;

		cudaEventRecord(start);
		TrivialGlobalSortIdx_1d<<<(M+1024-1)/1024, 1024>>>(M,d_idxnupts);
#ifdef SPREADTIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel TrivialGlobalSortIDx_1d \t\t%.3g ms\n",
			milliseconds);
#endif
	}
	return 0;
}

int CUSPREAD1D_NUPTSDRIVEN(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=d_plan->spopts.nspread;   // psi's support in terms of number of cells
	int pirange=d_plan->spopts.pirange;
	int *d_idxnupts=d_plan->idxnupts;
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	FLT sigma=d_plan->spopts.upsampfac;

	FLT* d_kx = d_plan->kx;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	cudaEventRecord(start);
	if(d_plan->opts.gpu_kerevalmeth){
		for(int t=0; t<blksize; t++){
			Spread_1d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx,
				d_c+t*M, d_fw+t*nf1, M, ns, nf1, sigma, d_idxnupts, pirange);
		}
	}else{
		for(int t=0; t<blksize; t++){
			Spread_1d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_c+t*M, 
				d_fw+t*nf1, M, ns, nf1, es_c, es_beta, d_idxnupts, pirange);
		}
	}

#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_1d_NUptsdriven (%d)\t%.3g ms\n",
		milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}
int CUSPREAD1D_SUBPROB_PROP(int nf1, int M, CUFINUFFT_PLAN d_plan)
/*
	This function determines the properties for spreading that are independent
	of the strength of the nodes,  only relates to the locations of the nodes,
	which only needs to be done once.
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;
	int bin_size_x=d_plan->opts.gpu_binsizex;
	if(bin_size_x < 0){
		cout<<"error: invalid binsize (binsizex) = (";
		cout<<bin_size_x<<")"<<endl;
		return 1; 
	}
	int numbins = ceil((FLT) nf1/bin_size_x);
#ifdef DEBUG
	cout<<"[debug  ] Dividing the uniform grids to bin size["
		<<d_plan->opts.gpu_binsizex<<"]"<<endl;
	cout<<"[debug  ] numbins = ["<<numbins<<"]"<<endl;
#endif

	FLT*   d_kx = d_plan->kx;

#ifdef DEBUG
	FLT *h_kx;
	h_kx = (FLT*)malloc(M*sizeof(FLT));

	checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		cout<<"[debug ]";
		cout <<"("<<setw(3)<<h_kx[i]<<")"<<endl;
	}
#endif
	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_sortidx = d_plan->sortidx;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;

	int *d_subprob_to_bin = NULL;

	int pirange=d_plan->spopts.pirange;

	cudaEventRecord(start);
	checkCudaErrors(cudaMemset(d_binsize,0,numbins*sizeof(int)));
	CalcBinSize_noghost_1d<<<(M+1024-1)/1024, 1024>>>(M,nf1,bin_size_x,
		numbins,d_binsize,d_kx,d_sortidx,pirange);
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcBinSize_noghost_1d \t\t%.3g ms\n",
		milliseconds);
#endif
#ifdef DEBUG
	int *h_binsize;// For debug
	h_binsize = (int*)malloc(numbins*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] bin size:"<<endl;
	cout<<"[debug ] ";
	for(int i=0; i<numbins; i++){
		if(i!=0) cout<<" ";
		cout <<"bin["<<setw(3)<<i<<"]="<<h_binsize[i];
	}
	free(h_binsize);
	cout<<"[debug ] ----------------------------------------------------"<<endl;
#endif
#ifdef DEBUG
	int *h_sortidx;
	h_sortidx = (int*)malloc(M*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
		cudaMemcpyDeviceToHost));
	cout<<"[debug ]";
	for(int i=0; i<M; i++){
		cout <<"[debug] point["<<setw(3)<<i<<"]="<<setw(3)<<h_sortidx[i]<<endl;
	}

#endif

	cudaEventRecord(start);
	int n=numbins;
	thrust::device_ptr<int> d_ptr(d_binsize);
	thrust::device_ptr<int> d_result(d_binstartpts);
	thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel BinStartPts_1d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int *h_binstartpts;
	h_binstartpts = (int*)malloc(numbins*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,numbins*sizeof(int),
				cudaMemcpyDeviceToHost));
	cout<<"[debug ] Result of scan bin_size array:"<<endl;
	cout<<"[debug ] ";
	for(int i=0; i<numbins; i++){
		if(i!=0) cout<<" ";
		cout <<"bin["<<setw(3)<<i<<"] = "<<setw(2)<<h_binstartpts[i];
	}
	free(h_binstartpts);
	cout<<"[debug ] ---------------------------------------------------"<<endl;
#endif
	cudaEventRecord(start);
	CalcInvertofGlobalSortIdx_1d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
		numbins,d_binstartpts,d_sortidx,d_kx,d_idxnupts,pirange,nf1);
#ifdef DEBUG
	int *h_idxnupts;
	h_idxnupts = (int*)malloc(M*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_idxnupts,d_idxnupts,M*sizeof(int),
				cudaMemcpyDeviceToHost));
	for (int i=0; i<M; i++){
		cout <<"[debug ] idx="<< h_idxnupts[i]<<endl;
	}
	free(h_idxnupts);
#endif
	cudaEventRecord(start);
	CalcSubProb_1d<<<(M+1024-1)/1024, 1024>>>(d_binsize,d_numsubprob,
		maxsubprobsize,numbins);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcSubProb_1d\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int* h_numsubprob;
	h_numsubprob = (int*) malloc(n*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_numsubprob,d_numsubprob,numbins*
				sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] ";
	for(int i=0; i<numbins; i++){
		if(i!=0) cout<<" ";
		cout <<"nsub["<<setw(3)<<i<<"] = "<<setw(2)<<h_numsubprob[i];
	}
	cout << endl;
	free(h_numsubprob);
#endif
	d_ptr    = thrust::device_pointer_cast(d_numsubprob);
	d_result = thrust::device_pointer_cast(d_subprobstartpts+1);
	thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
	checkCudaErrors(cudaMemset(d_subprobstartpts,0,sizeof(int)));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Scan Subprob array\t\t%.3g ms\n", milliseconds);
#endif

#ifdef DEBUG
	printf("[debug ] Subproblem start points\n");
	int* h_subprobstartpts;
	h_subprobstartpts = (int*) malloc((n+1)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_subprobstartpts,d_subprobstartpts,
				(n+1)*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] ";
	for(int i=0; i<numbins; i++){
		if(i!=0) cout<<" ";
		cout <<"nsub["<<setw(3)<<i<<"] = "<<setw(2)<<h_subprobstartpts[i];
	}
	cout << endl;
	printf("[debug ] Total number of subproblems = %d\n", h_subprobstartpts[n]);
	free(h_subprobstartpts);
#endif
	cudaEventRecord(start);
	int totalnumsubprob;
	checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[n],
		sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));
	MapBintoSubProb_1d<<<(numbins+1024-1)/1024, 1024>>>(
			d_subprob_to_bin,d_subprobstartpts,d_numsubprob,numbins);
	assert(d_subprob_to_bin != NULL);
	if (d_plan->subprob_to_bin != NULL) cudaFree(d_plan->subprob_to_bin);
	d_plan->subprob_to_bin = d_subprob_to_bin;
	assert(d_plan->subprob_to_bin != NULL);
	d_plan->totalnumsubprob = totalnumsubprob;
#ifdef DEBUG
	printf("[debug ] Map Subproblem to Bins\n");
	int* h_subprob_to_bin;
	h_subprob_to_bin = (int*) malloc((totalnumsubprob)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_subprob_to_bin,d_subprob_to_bin,
				(totalnumsubprob)*sizeof(int),cudaMemcpyDeviceToHost));
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
	return 0;
}

int CUSPREAD1D_SUBPROB(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ns=d_plan->spopts.nspread;// psi's support in terms of number of cells
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;

	// assume that bin_size_x > ns/2;
	int bin_size_x=d_plan->opts.gpu_binsizex;
	int numbins = ceil((FLT) nf1/bin_size_x);

	FLT* d_kx = d_plan->kx;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;

	int totalnumsubprob=d_plan->totalnumsubprob;
	int *d_subprob_to_bin = d_plan->subprob_to_bin;

	int pirange=d_plan->spopts.pirange;

	FLT sigma=d_plan->opts.upsampfac;
	cudaEventRecord(start);

	size_t sharedplanorysize = (bin_size_x+2*(int)ceil(ns/2.0))*sizeof(CUCPX);
	if(sharedplanorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	if(d_plan->opts.gpu_kerevalmeth){
		for(int t=0; t<blksize; t++){
			Spread_1d_Subprob_Horner<<<totalnumsubprob, 256,
				sharedplanorysize>>>(d_kx, d_c+t*M, d_fw+t*nf1, M,
				ns, nf1, sigma, d_binstartpts, d_binsize, bin_size_x,
				d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
				numbins, d_idxnupts, pirange);
		}
	}else{
		for(int t=0; t<blksize; t++){
			Spread_1d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
				d_kx, d_c+t*M, d_fw+t*nf1, M, ns, nf1, es_c, es_beta, sigma, 
				d_binstartpts, d_binsize, bin_size_x, d_subprob_to_bin, 
				d_subprobstartpts, d_numsubprob, maxsubprobsize, numbins,
				d_idxnupts, pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_1d_Subprob (%d)\t\t%.3g ms\n",
		milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}
