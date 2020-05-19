#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include <cuspreadinterp.h>
#include <memtransfer.h>

using namespace std;

int cufinufft_spread2d(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, 
	const FLT *h_kx, const FLT *h_ky, const CPX *h_c, cufinufft_plan* d_plan)
/*
	This c function is written for only doing 2D spreading. It includes 
	allocating, transfering, and freeing the memories on gpu. See 
	test/spread_2d.cu for usage.

	Melody Shih 07/25/19
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->c,M*sizeof(CUCPX)));

	int ier;
	//int ier = setup_spreader_for_nufft(d_plan->spopts, eps, d_plan->opts);
	d_plan->ms = ms;
	d_plan->mt = mt;
	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->M = M;
	d_plan->ntransfcufftplan = 1;

	cudaEventRecord(start);
	ier = allocgpumem2d_plan(d_plan);
	ier = allocgpumem2d_nupts(d_plan);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Allocate GPU memory\t %.3g ms\n", milliseconds);
#endif

	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->kx,h_kx,M*sizeof(FLT),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->ky,h_ky,M*sizeof(FLT),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->c, h_c, M*sizeof(CUCPX),
		cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory HtoD (%d Bytes) \t%.3g ms\n", 
		2*M*sizeof(FLT)+M*sizeof(CUCPX), milliseconds);
#endif

	if(d_plan->opts.gpu_method == 2){
		ier = cuspread2d_subprob_prop(nf1,nf2,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread2d_subprob_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}

	if(d_plan->opts.gpu_method == 3){
		ier = cuspread2d_paul_prop(nf1,nf2,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread2d_subprob_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}

	cudaEventRecord(start);
	ier = cuspread2d(d_plan,1);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Spread (%d)\t\t %.3g ms\n", d_plan->opts.gpu_method, 
		milliseconds);
#endif
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(h_fw,d_plan->fw,nf1*nf2*sizeof(CUCPX),
		cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory DtoH (%d Bytes) \t %.3g ms\n",
		nf1*nf2*sizeof(CUCPX),  milliseconds);
#endif
	cudaEventRecord(start);
	freegpumemory2d(d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	cudaFree(d_plan->kx);
	cudaFree(d_plan->ky);
	cudaFree(d_plan->c);
	return ier;
}

int cuspread2d(cufinufft_plan* d_plan, int blksize)
/*
	A wrapper for different spreading methods. 

	Methods available:
	(1) Non-uniform points driven
	(2) Subproblem
	(3) Paul

	Melody Shih 07/25/19
*/
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
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
				ier = cuspread2d_nuptsdriven(nf1, nf2, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_nuptsdriven"<<endl;
					return 1;
				}
			}
			break;
		case 2:
			{
				cudaEventRecord(start);
				ier = cuspread2d_subprob(nf1, nf2, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_subprob"<<endl;
					return 1;
				}
			}
			break;
		case 3:
			{
				cudaEventRecord(start);
				ier = cuspread2d_paul(nf1, nf2, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_paul"<<endl;
					return 1;
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1,2,3"<<endl;
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

int cuspread2d_nuptsdriven_prop(int nf1, int nf2, int M, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if(d_plan->opts.gpu_sort){
		dim3 threadsPerBlock;
		dim3 blocks;

		int bin_size_x=d_plan->opts.gpu_binsizex;
		int bin_size_y=d_plan->opts.gpu_binsizey;
		int numbins[2];
		numbins[0] = ceil((FLT) nf1/bin_size_x);
		numbins[1] = ceil((FLT) nf2/bin_size_y);

#ifdef DEBUG
		cout<<"[debug ] Dividing the uniform grids to bin size["
			<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"]"<<endl;
		cout<<"[debug ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif

		FLT*   d_kx = d_plan->kx;
		FLT*   d_ky = d_plan->ky;
#ifdef DEBUG
		FLT *h_kx;
		FLT *h_ky;
		h_kx = (FLT*)malloc(M*sizeof(FLT));
		h_ky = (FLT*)malloc(M*sizeof(FLT));

		checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),
			cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_ky,d_ky,M*sizeof(FLT),
			cudaMemcpyDeviceToHost));
		for(int i=M-10; i<M; i++){
			cout<<"[debug ] ";
			cout <<"("<<setw(3)<<h_kx[i]<<","<<setw(3)<<h_ky[i]<<")"<<endl;
		}
#endif
		int *d_binsize = d_plan->binsize;
		int *d_binstartpts = d_plan->binstartpts;
		int *d_sortidx = d_plan->sortidx;
		int *d_idxnupts = d_plan->idxnupts;
		void *d_temp_storage = NULL;

		int pirange = d_plan->spopts.pirange;

		cudaEventRecord(start);
		checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*
			sizeof(int)));
		CalcBinSize_noghost_2d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,
			bin_size_x,bin_size_y,numbins[0],numbins[1],
			d_binsize,d_kx,d_ky,d_sortidx,pirange);
#ifdef SPREADTIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel CalcBinSize_noghost_2d \t\t%.3g ms\n", 
			milliseconds);
#endif
#ifdef DEBUG
		int *h_binsize;// For debug
		h_binsize     = (int*)malloc(numbins[0]*numbins[1]*sizeof(int));
		checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*
			sizeof(int),cudaMemcpyDeviceToHost));
		cout<<"[debug ] bin size:"<<endl;
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<" bin["<<setw(1)<<i<<","<<setw(1)<<j<<"]="<<
					h_binsize[i+j*numbins[0]];
			}
				cout<<endl;
		}
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
				printf("(%10.10f, %10.10f) ", RESCALE(h_kx[i],nf1,pirange),
                                     RESCALE(h_ky[i],nf1,pirange)); 
				printf("(%10.10f, %10.10f) ", RESCALE(h_kx[i],nf1,pirange)/32,
                                     RESCALE(h_ky[i],nf1,pirange)/32); 
				printf("(%f, %f)\n", floor(RESCALE(h_kx[i],nf1,pirange)/32),
                                     floor(RESCALE(h_ky[i],nf1,pirange)/32)); 
			}
		}
#endif
		cudaEventRecord(start);
		int n=numbins[0]*numbins[1];
		size_t temp_storage_bytes = 0;
		assert(d_temp_storage == NULL);
		CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,
			temp_storage_bytes,d_binsize, d_binstartpts, n));
		// Allocate temporary storage for inclusive prefix scan
		checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,
			temp_storage_bytes,d_binsize, d_binstartpts, n));
#ifdef SPREADTIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel BinStartPts_2d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
		int *h_binstartpts;
		h_binstartpts = (int*)malloc((numbins[0]*numbins[1])*sizeof(int));
		checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]*
			numbins[1])*sizeof(int),cudaMemcpyDeviceToHost));
		cout<<"[debug ] Result of scan bin_size array:"<<endl;
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<" bin["<<setw(1)<<i<<","<<setw(1)<<j<<"]="<<
					h_binstartpts[i+j*numbins[0]];
			}
			cout<<endl;
		}
		free(h_binstartpts);
		cout<<"[debug ] ------------------------------------------------"<<endl;
#endif
		cudaEventRecord(start);
		CalcInvertofGlobalSortIdx_2d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
			bin_size_y,numbins[0],numbins[1],d_binstartpts,d_sortidx,d_kx,d_ky,
			d_idxnupts,pirange,nf1,nf2);
#ifdef SPREADTIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel CalcInvertofGlobalSortIdx_2d \t%.3g ms\n", 
			milliseconds);
#endif
#ifdef DEBUG
		int *h_idxnupts;
		h_idxnupts = (int*)malloc(M*sizeof(int));
		checkCudaErrors(cudaMemcpy(h_idxnupts,d_idxnupts,M*sizeof(int),
					cudaMemcpyDeviceToHost));
		for (int i=0; i<10; i++){
			cout <<"[debug ] idx="<< h_idxnupts[i]<<endl;
		}
		free(h_idxnupts);
#endif
		cudaFree(d_temp_storage);
	}else{
		int *d_idxnupts = d_plan->idxnupts;

		cudaEventRecord(start);
		TrivialGlobalSortIdx_2d<<<(M+1024-1)/1024, 1024>>>(M,d_idxnupts);
#ifdef SPREADTIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel TrivialGlobalSortIDx_2d \t\t%.3g ms\n", 
			milliseconds);
#endif
	}
	return 0;
}

int cuspread2d_nuptsdriven(int nf1, int nf2, int M, cufinufft_plan *d_plan,
	int blksize)
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
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	cudaEventRecord(start);
	if(d_plan->opts.gpu_kerevalmeth){
		for(int t=0; t<blksize; t++){
			Spread_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, 
				d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, sigma, 
				d_idxnupts, pirange);
		}
	}else{
		for(int t=0; t<blksize; t++){
			Spread_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, 
				d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, es_c, es_beta, 
				d_idxnupts, pirange);
		}
	}

#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_2d_NUptsdriven (%d)\t%.3g ms\n", 
		milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}
int cuspread2d_subprob_prop(int nf1, int nf2, int M, cufinufft_plan *d_plan)
/* 
	This function determines the properties for spreading that are independent
	of the strength of the nodes,  only relates to the locations of the nodes, 
	which only needs to be done once.
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;
	int bin_size_x=d_plan->opts.gpu_binsizex;
	int bin_size_y=d_plan->opts.gpu_binsizey;
	int numbins[2];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef DEBUG
	cout<<"[debug  ] Dividing the uniform grids to bin size["
		<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"]"<<endl;
	cout<<"[debug  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif

	FLT*   d_kx = d_plan->kx;
	FLT*   d_ky = d_plan->ky;

#ifdef DEBUG
	FLT *h_kx;
	FLT *h_ky;
	h_kx = (FLT*)malloc(M*sizeof(FLT));
	h_ky = (FLT*)malloc(M*sizeof(FLT));

	checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_ky,d_ky,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		cout<<"[debug ]";
		cout <<"("<<setw(3)<<h_kx[i]<<","<<setw(3)<<h_ky[i]<<")"<<endl;
	}
#endif
	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_sortidx = d_plan->sortidx;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;

	int *d_subprob_to_bin = NULL;
	void *d_temp_storage = NULL;

	int pirange=d_plan->spopts.pirange;

	cudaEventRecord(start);
	checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*sizeof(int)));
	CalcBinSize_noghost_2d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,bin_size_x,
		bin_size_y,numbins[0],numbins[1],d_binsize,d_kx,d_ky,d_sortidx,pirange);
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcBinSize_noghost_2d \t\t%.3g ms\n", 
		milliseconds);
#endif
#ifdef DEBUG
	int *h_binsize;// For debug
	h_binsize     = (int*)malloc(numbins[0]*numbins[1]*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*
		sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] bin size:"<<endl;
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<" bin["<<setw(3)<<i<<","<<setw(3)<<j<<"]="<<
				h_binsize[i+j*numbins[0]];
		}
		cout<<endl;
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
		cout <<"point["<<setw(3)<<i<<"]="<<setw(3)<<h_sortidx[i]<<endl;
	}

#endif

	cudaEventRecord(start);
	int n=numbins[0]*numbins[1];
	size_t temp_storage_bytes = 0;
	assert(d_temp_storage == NULL);
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, 
				temp_storage_bytes, 
				d_binsize, d_binstartpts, 
				n));
	// Allocate temporary storage for inclusive prefix scan
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes)); 
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, 
				temp_storage_bytes, 
				d_binsize, d_binstartpts, 
				n));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel BinStartPts_2d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int *h_binstartpts;
	h_binstartpts = (int*)malloc((numbins[0]*numbins[1])*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,
				(numbins[0]*numbins[1])*sizeof(int),
				cudaMemcpyDeviceToHost));
	cout<<"[debug ] Result of scan bin_size array:"<<endl;
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<"bin["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)
				<<h_binstartpts[i+j*numbins[0]];
		}
		cout<<endl;
	}
	free(h_binstartpts);
	cout<<"[debug ] ---------------------------------------------------"<<endl;
#endif
	cudaEventRecord(start);
	CalcInvertofGlobalSortIdx_2d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
		bin_size_y,numbins[0],numbins[1],d_binstartpts,d_sortidx,d_kx,d_ky,
		d_idxnupts,pirange,nf1,nf2);
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
	CalcSubProb_2d<<<(M+1024-1)/1024, 1024>>>(d_binsize,d_numsubprob,
		maxsubprobsize,numbins[0]*numbins[1]);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcSubProb_2d\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int* h_numsubprob;
	h_numsubprob = (int*) malloc(n*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_numsubprob,d_numsubprob,numbins[0]*numbins[1]*
				sizeof(int),cudaMemcpyDeviceToHost));
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<"nsub["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)<<
				h_numsubprob[i+j*numbins[0]];
		}
		cout<<endl;
	}
	free(h_numsubprob);
#endif
	// Scanning the same length array, so we don't need calculate 
	// temp_storage_bytes here
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, 
				temp_storage_bytes, d_numsubprob, d_subprobstartpts+1, n));
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
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<"nsub["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<setw(2)<<
				h_subprobstartpts[i+j*numbins[0]];
		}
		cout<<endl;
	}
	printf("[debug ] Total number of subproblems = %d\n", h_subprobstartpts[n]);
	free(h_subprobstartpts);
#endif
	int totalnumsubprob;
	checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[n],
		sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));
	MapBintoSubProb_2d<<<(numbins[0]*numbins[1]+1024-1)/1024, 1024>>>(
			d_subprob_to_bin,d_subprobstartpts,d_numsubprob,numbins[0]*numbins[1]);
	assert(d_subprob_to_bin != NULL);
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
	cudaFree(d_temp_storage);
	return 0;
}

int cuspread2d_subprob(int nf1, int nf2, int M, cufinufft_plan *d_plan, 
	int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=d_plan->spopts.nspread;// psi's support in terms of number of cells
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;

	// assume that bin_size_x > ns/2;
	int bin_size_x=d_plan->opts.gpu_binsizex;
	int bin_size_y=d_plan->opts.gpu_binsizey;
	int numbins[2];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
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

	size_t sharedplanorysize = (bin_size_x+2*(int)ceil(ns/2.0))*
							   (bin_size_y+2*(int)ceil(ns/2.0))*
							   sizeof(CUCPX);
	if(sharedplanorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	if(d_plan->opts.gpu_kerevalmeth){
		for(int t=0; t<blksize; t++){
			Spread_2d_Subprob_Horner<<<totalnumsubprob, 256, 
				sharedplanorysize>>>(d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, 
				ns, nf1, nf2, sigma, d_binstartpts, d_binsize, bin_size_x, 
				bin_size_y, d_subprob_to_bin, d_subprobstartpts, 
				d_numsubprob, maxsubprobsize,numbins[0],numbins[1], 
				d_idxnupts, pirange);
		}
	}else{
		for(int t=0; t<blksize; t++){
			Spread_2d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
				d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, 
				es_c, es_beta, sigma,d_binstartpts, d_binsize, bin_size_x, 
				bin_size_y, d_subprob_to_bin, d_subprobstartpts,
				d_numsubprob, maxsubprobsize, numbins[0], numbins[1],
				d_idxnupts, pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_2d_Subprob (%d)\t\t%.3g ms\n", 
		milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}
