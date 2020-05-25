#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "../cuspreadinterp.h"
#include "../memtransfer.h"

using namespace std;

#ifdef DEBUG
static
int CalcGlobalIdx(int xidx, int yidx, int zidx, int onx, int ony, int onz,
	int bnx, int bny, int bnz){
	int oix,oiy,oiz;
	oix = xidx/bnx;
	oiy = yidx/bny;
	oiz = zidx/bnz;
	return   (oix+oiy*onx + oiz*onx*ony)*(bnx*bny*bnz) +
			 (xidx%bnx+yidx%bny*bnx+zidx%bnz*bny*bnx);
}
#endif

// This is a function only doing spread includes device memory allocation, transfer, free
int cufinufft_spread3d(int ms, int mt, int mu, int nf1, int nf2, int nf3,
	CPX* h_fw, int M, const FLT *h_kx, const FLT *h_ky, const FLT* h_kz,
	const CPX *h_c, FLT eps, cufinufft_plan* d_plan)
/*
	This c function is written for only doing 3D spreading. It includes 
	allocating, transfering, and freeing the memories on gpu. See 
	test/spread_3d.cu for usage.

	Melody Shih 07/25/19
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	//ier = setup_spreader_for_nufft(d_plan->spopts, eps, d_plan->opts);
	d_plan->ms = ms;
	d_plan->mt = mt;
	d_plan->mu = mu;
	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->nf3 = nf3;
	d_plan->M = M;
	d_plan->ntransfcufftplan = 1;

	cudaEventRecord(start);
	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->kz,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->c,M*sizeof(CUCPX)));

	ier = allocgpumem3d_plan(d_plan);
	ier = allocgpumem3d_nupts(d_plan);
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
	checkCudaErrors(cudaMemcpy(d_plan->kz,h_kz,M*sizeof(FLT),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_plan->c, h_c, M*sizeof(CUCPX),
		cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory HtoD \t %.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	if(d_plan->opts.gpu_method == 1){
		ier = cuspread3d_nuptsdriven_prop(nf1,nf2,nf3,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}
	if(d_plan->opts.gpu_method == 2){
		ier = cuspread3d_subprob_prop(nf1,nf2,nf3,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d_subprob_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}
	if(d_plan->opts.gpu_method == 4){
		ier = cuspread3d_blockgather_prop(nf1,nf2,nf3,M,d_plan);
		if(ier != 0 ){
			printf("error: cuspread3d_blockgather_prop, method(%d)\n", 
				d_plan->opts.gpu_method);
			return 0;
		}
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Obtain SubProb prop\t %.3g ms\n", milliseconds);
#endif

	cudaEventRecord(start);
	ier = cuspread3d(d_plan, 1);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Spread (%d)\t\t %.3g ms\n", d_plan->opts.gpu_method, milliseconds);
#endif
	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(h_fw,d_plan->fw,nf1*nf2*nf3*sizeof(CUCPX),
		cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Copy memory DtoH\t %.3g ms\n", milliseconds);
#endif

	cudaEventRecord(start);
	freegpumemory3d(d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	cudaFree(d_plan->kx);
	cudaFree(d_plan->ky);
	cudaFree(d_plan->kz);
	cudaFree(d_plan->c);
	return ier;
}

int cuspread3d(cufinufft_plan* d_plan, int blksize)
/*
	A wrapper for different spreading methods. 

	Methods available:
	(1) Non-uniform points driven
	(2) Subproblem
	(4) Block gather

	Melody Shih 07/25/19
*/
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int M = d_plan->M;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier = 0;
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				cudaEventRecord(start);
				ier = cuspread3d_nuptsdriven(nf1, nf2, nf3, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cnufftspread3d_gpu_subprob"<<endl;
					return 1;
				}
			}
			break;
		case 2:
			{
				cudaEventRecord(start);
				ier = cuspread3d_subprob(nf1, nf2, nf3, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cnufftspread3d_gpu_subprob"<<endl;
					return 1;
				}
			}
			break;
		case 4:
			{
				cudaEventRecord(start);
				ier = cuspread3d_blockgather(nf1, nf2, nf3, M, d_plan, blksize);
				if(ier != 0 ){
					cout<<"error: cnufftspread3d_gpu_subprob"<<endl;
					return 1;
				}
			}
			break;
		default:
			cerr<<"error: incorrect method, should be 1,2,4"<<endl;
			return 2;
	}
	return ier;
}

int cuspread3d_nuptsdriven_prop(int nf1, int nf2, int nf3, int M, 
	cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if(d_plan->opts.gpu_sort){
		dim3 threadsPerBlock;
		dim3 blocks;

		int bin_size_x=d_plan->opts.gpu_binsizex;
		int bin_size_y=d_plan->opts.gpu_binsizey;
		int bin_size_z=d_plan->opts.gpu_binsizez;
		int numbins[3];
		numbins[0] = ceil((FLT) nf1/bin_size_x);
		numbins[1] = ceil((FLT) nf2/bin_size_y);
		numbins[2] = ceil((FLT) nf2/bin_size_z);

#ifdef DEBUG
		cout<<"[debug ] Dividing the uniform grids to bin size["
			<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"x"<<
			d_plan->opts.gpu_binsizez<<"]"<<endl;
		cout<<"[debug ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"x"<<numbins[2]
			<<"]"<<endl;
#endif

		FLT*   d_kx = d_plan->kx;
		FLT*   d_ky = d_plan->ky;
		FLT*   d_kz = d_plan->kz;
#ifdef DEBUG
		FLT *h_kx;
		FLT *h_ky;
		FLT *h_kz;
		h_kx = (FLT*)malloc(M*sizeof(FLT));
		h_ky = (FLT*)malloc(M*sizeof(FLT));
		h_kz = (FLT*)malloc(M*sizeof(FLT));

		checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),
			cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_ky,d_ky,M*sizeof(FLT),
			cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_kz,d_kz,M*sizeof(FLT),
			cudaMemcpyDeviceToHost));
		for(int i=0; i<10; i++){
			cout<<"[debug ] ";
			cout <<"("<<setw(3)<<h_kx[i]<<","<<setw(3)<<h_ky[i]<<","<<setw(3)
				<<h_kz[i]<<")"<<endl;
		}
#endif

		int *d_binsize = d_plan->binsize;
		int *d_binstartpts = d_plan->binstartpts;
		int *d_sortidx = d_plan->sortidx;
		int *d_idxnupts = d_plan->idxnupts;
		void *d_temp_storage = NULL;

		int pirange = d_plan->spopts.pirange;

		cudaEventRecord(start);
		checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*numbins[2]*
			sizeof(int)));
		CalcBinSize_noghost_3d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,nf3,
			bin_size_x,bin_size_y,bin_size_z,numbins[0],numbins[1],numbins[2],
			d_binsize,d_kx,d_ky,d_kz,d_sortidx,pirange);
#ifdef SPREADTIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel CalcBinSize_noghost_3d \t\t%.3g ms\n", 
			milliseconds);
#endif
#ifdef DEBUG
		int *h_binsize;// For debug
		h_binsize     = (int*)malloc(numbins[0]*numbins[1]*numbins[2]*
			sizeof(int));
		checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*
			numbins[2]*sizeof(int),cudaMemcpyDeviceToHost));
		cout<<"[debug ] bin size:"<<endl;
		for(int k=0; k<numbins[2]; k++){
			for(int j=0; j<numbins[1]; j++){
				cout<<"[debug ] ";
				for(int i=0; i<numbins[0]; i++){
					if(i!=0) cout<<" ";
					cout <<" bin["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)
						<<k<<"]="<<h_binsize[i+j*numbins[0]+k*numbins[0]*
						numbins[1]];
				}
				cout<<endl;
			}
		}
		free(h_binsize);
		cout<<"[debug ] ------------------------------------------------"<<endl;
#endif
#ifdef DEBUG
		int *h_sortidx;
		h_sortidx = (int*)malloc(M*sizeof(int));

		checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
			cudaMemcpyDeviceToHost));
		for(int i=0; i<M; i++){
			cout<<"[debug ] ";
			cout <<"point["<<setw(3)<<i<<"]="<<setw(3)<<h_sortidx[i]<<endl;
		}
#endif

		cudaEventRecord(start);
		int n=numbins[0]*numbins[1]*numbins[2];
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
		printf("[time  ] \tKernel BinStartPts_3d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
		int *h_binstartpts;
		h_binstartpts = (int*)malloc((numbins[0]*numbins[1]*numbins[2])*
			sizeof(int));
		checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]*
			numbins[1]*numbins[2])*sizeof(int),cudaMemcpyDeviceToHost));
		cout<<"[debug ] Result of scan bin_size array:"<<endl;
		for(int k=0; k<numbins[2]; k++){
			for(int j=0; j<numbins[1]; j++){
				cout<<"[debug ] ";
				for(int i=0; i<numbins[0]; i++){
					if(i!=0) cout<<" ";
					cout <<" bin["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<
						k<<"]="<<h_binstartpts[i+j*numbins[0]+k*numbins[0]*numbins[1]];
				}
				cout<<endl;
			}
		}
		free(h_binstartpts);
		cout<<"[debug ] ------------------------------------------------"<<endl;
#endif
		cudaEventRecord(start);
		CalcInvertofGlobalSortIdx_3d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
			bin_size_y,bin_size_z,numbins[0],numbins[1],numbins[2],d_binstartpts,
			d_sortidx,d_kx,d_ky,d_kz,d_idxnupts, pirange, nf1, nf2, nf3);
#ifdef SPREADTIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel CalcInvertofGlobalSortIdx_3d \t%.3g ms\n", 
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
		TrivialGlobalSortIdx_3d<<<(M+1024-1)/1024, 1024>>>(M,d_idxnupts);
#ifdef SPREADTIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tKernel TrivialGlobalSortIDx_3d \t\t%.3g ms\n", 
			milliseconds);
#endif
	}
	return 0;
}

int cuspread3d_nuptsdriven(int nf1, int nf2, int nf3, int M, 
	cufinufft_plan *d_plan, int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=d_plan->spopts.nspread;   // psi's support in terms of number of cells
	FLT sigma=d_plan->spopts.upsampfac;
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	int pirange=d_plan->spopts.pirange;

	int* d_idxnupts = d_plan->idxnupts;
	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	FLT* d_kz = d_plan->kz;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	cudaEventRecord(start);
	if(d_plan->opts.gpu_kerevalmeth==1){
		for(int t=0; t<blksize; t++){
			Spread_3d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, d_ky, 
				d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, M, ns, nf1, nf2, nf3, sigma, 
				d_idxnupts,pirange);
		}
	}else{
		for(int t=0; t<blksize; t++){
			Spread_3d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_kz,
				d_c+t*M, d_fw+t*nf1*nf2*nf3, M, ns, nf1, nf2, nf3, es_c, es_beta, 
				d_idxnupts,pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_3d_NUptsdriven (%d)\t%.3g ms\n", 
		milliseconds,d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}

int cuspread3d_blockgather_prop(int nf1, int nf2, int nf3, int M, 
	cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int pirange = d_plan->spopts.pirange;

	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;
	int o_bin_size_x = d_plan->opts.gpu_obinsizex;
	int o_bin_size_y = d_plan->opts.gpu_obinsizey;
	int o_bin_size_z = d_plan->opts.gpu_obinsizez;
	int numobins[3];

	numobins[0] = ceil((FLT) nf1/o_bin_size_x);
	numobins[1] = ceil((FLT) nf2/o_bin_size_y);
	numobins[2] = ceil((FLT) nf3/o_bin_size_z);

	int bin_size_x=d_plan->opts.gpu_binsizex;
	int bin_size_y=d_plan->opts.gpu_binsizey;
	int bin_size_z=d_plan->opts.gpu_binsizez;
	int binsperobinx, binsperobiny, binsperobinz;
	int numbins[3];
	binsperobinx = o_bin_size_x/bin_size_x+2;
	binsperobiny = o_bin_size_y/bin_size_y+2;
	binsperobinz = o_bin_size_z/bin_size_z+2;
	numbins[0] = numobins[0]*(binsperobinx);
	numbins[1] = numobins[1]*(binsperobiny);
	numbins[2] = numobins[2]*(binsperobinz);
#ifdef DEBUG
	cout<<"[debug ] Dividing the uniform grids to bin size["
		<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"x"<<
		d_plan->opts.gpu_binsizez<<"]"<<endl;
	cout<<"[debug ] numobins = ["<<numobins[0]<<"x"<<numobins[1]<<"x"<<
		numobins[2]<<"]"<<endl;
	cout<<"[debug ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"x"<<
		numbins[2]<<"]"<<endl;
#endif

	FLT*   d_kx = d_plan->kx;
	FLT*   d_ky = d_plan->ky;
	FLT*   d_kz = d_plan->kz;

#ifdef DEBUG
	FLT *h_kx, *h_ky, *h_kz;
	h_kx = (FLT*)malloc(M*sizeof(FLT));
	h_ky = (FLT*)malloc(M*sizeof(FLT));
	h_kz = (FLT*)malloc(M*sizeof(FLT));

	checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_ky,d_ky,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_kz,d_kz,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		cout<<"[debug ] ";
		cout <<"("<<setw(3)<<h_kx[i]<<","<<setw(3)<<h_ky[i]<<","<<h_kz[i]<<")"
			<<endl;
	}
#endif
	int *d_binsize = d_plan->binsize;
	int *d_sortidx = d_plan->sortidx;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_numsubprob = d_plan->numsubprob;
	void*d_temp_storage = NULL;
	int *d_idxnupts = NULL;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_subprob_to_bin = NULL;

	cudaEventRecord(start);
	checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*numbins[2]*
		sizeof(int)));
	LocateNUptstoBins_ghost<<<(M+1024-1)/1024, 1024>>>(M,bin_size_x,
		bin_size_y,bin_size_z,numobins[0],numobins[1],numobins[2],binsperobinx,
		binsperobiny, binsperobinz,d_binsize,d_kx,
		d_ky,d_kz,d_sortidx,pirange,nf1,nf2,nf3);
#ifdef DEBUG
	threadsPerBlock.x=8;
	threadsPerBlock.y=8;
	threadsPerBlock.z=8;
	blocks.x = (threadsPerBlock.x+numbins[0]-1)/threadsPerBlock.x;
	blocks.y = (threadsPerBlock.y+numbins[1]-1)/threadsPerBlock.y;
	blocks.z = (threadsPerBlock.z+numbins[2]-1)/threadsPerBlock.z;

	Temp<<<blocks, threadsPerBlock>>>(binsperobinx, binsperobiny, binsperobinz,
		numobins[0], numobins[1], numobins[2], d_binsize);
#endif
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel LocateNUptstoBins_ghost \t\t%.3g ms\n",
		milliseconds);
#endif
#ifdef DEBUG
	int *h_binsize;// For debug
	h_binsize     = (int*)malloc(numbins[0]*numbins[1]*numbins[2]*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*
		numbins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] bin size:"<<endl;
	for(int k=0; k<numbins[2]; k++){
		cout<<"[debug ]"<<endl;
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i%binsperobiny == 0 && i!=0)
				cout<<"|";
				int binidx = CalcGlobalIdx(i,j,k,numobins[0],numobins[1],
					numobins[2],binsperobinx,binsperobiny,binsperobinz);
				if(i!=0) cout<<" ";
				cout<<" b["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k
					<<"]= "<<setw(3)<<h_binsize[binidx];
			}
			cout<<endl;
		}
	}
	cout<<"[debug ] ---------------------------------------------------"<<endl;
#endif
#ifdef DEBUG
	int *h_sortidx;
	h_sortidx = (int*)malloc(M*sizeof(int));

	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
		cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		cout <<"[debug ] point["<<setw(3)<<i<<"]="<<setw(3)<<h_sortidx[i]<<endl;
	}
#endif
	cudaEventRecord(start);
	threadsPerBlock.x=8;
	threadsPerBlock.y=8;
	threadsPerBlock.z=8;

	blocks.x = (threadsPerBlock.x+numbins[0]-1)/threadsPerBlock.x;
	blocks.y = (threadsPerBlock.y+numbins[1]-1)/threadsPerBlock.y;
	blocks.z = (threadsPerBlock.z+numbins[2]-1)/threadsPerBlock.z;

	FillGhostBins<<<blocks, threadsPerBlock>>>(binsperobinx, binsperobiny,
		binsperobinz, numobins[0], numobins[1], numobins[2], d_binsize);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel FillGhostBins \t\t\t%.3g ms\n",
		milliseconds);
#endif
#ifdef DEBUG
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*
		numbins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] Filled ghost bins:"<<endl;
	for(int k=0; k<numbins[2]; k++){
		cout<<"[debug ] "<<endl;
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				int binidx = CalcGlobalIdx(i,j,k,numobins[0],numobins[1],
					numobins[2],binsperobinx,binsperobiny,binsperobinz);
				cout<<" b["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k
					<<"]= "<<setw(3)<<h_binsize[binidx];
			}
			cout<<endl;
		}
	}
	cout<<"[debug ] ---------------------------------------------------"<<endl;
#endif
	cudaEventRecord(start);
	int n=numbins[0]*numbins[1]*numbins[2];
	size_t temp_storage_bytes = 0;
	assert(d_temp_storage == NULL);
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage,
		temp_storage_bytes, d_binsize, d_binstartpts, n));
	// Allocate temporary storage for inclusive prefix scan
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage,
		temp_storage_bytes, d_binsize, d_binstartpts+1, n));
	checkCudaErrors(cudaMemset(d_binstartpts,0,sizeof(int)));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel BinStartPts_3d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int *h_binstartpts;
	h_binstartpts = (int*)malloc((numbins[0]*numbins[1]*numbins[2])*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]*
		numbins[1]*numbins[2])*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] Result of scan bin_size array:"<<endl;
	for(int k=0; k<numbins[2]; k++){
		cout<<"[debug ] "<<endl;
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				int binidx = CalcGlobalIdx(i,j,k,numobins[0],numobins[1],
					numobins[2],binsperobinx,binsperobiny,binsperobinz);
				cout<<" b["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k
					<<"]= "<<setw(3)<<h_binstartpts[binidx];
			}
			cout<<endl;
		}
	}
	cout<<"[debug ] ----------------------------------------------------"<<endl;
#endif
	cudaEventRecord(start);
	int totalNUpts;
	checkCudaErrors(cudaMemcpy(&totalNUpts,&d_binstartpts[n],
		sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMalloc(&d_idxnupts,totalNUpts*sizeof(int)));
	cudaEventRecord(start);
	CalcInvertofGlobalSortIdx_ghost<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
		bin_size_y,bin_size_z,numobins[0],numobins[1],numobins[2],binsperobinx,
		binsperobiny,binsperobinz,d_binstartpts,d_sortidx,d_kx,d_ky,d_kz,
		d_idxnupts,pirange,nf1,nf2,nf3);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcInvertofGlobalIdx_ghost \t%.3g ms\n",
		milliseconds);
#endif
	cudaEventRecord(start);
	threadsPerBlock.x=2;
	threadsPerBlock.y=2;
	threadsPerBlock.z=2;

	blocks.x = (threadsPerBlock.x+numbins[0]-1)/threadsPerBlock.x;
	blocks.y = (threadsPerBlock.y+numbins[1]-1)/threadsPerBlock.y;
	blocks.z = (threadsPerBlock.z+numbins[2]-1)/threadsPerBlock.z;

	GhostBinPtsIdx<<<blocks, threadsPerBlock>>>(binsperobinx, binsperobiny,
		binsperobinz, numobins[0], numobins[1], numobins[2], d_binsize,
		d_idxnupts, d_binstartpts, M);
	d_plan->idxnupts = d_idxnupts;
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel GhostBinPtsIdx \t\t\t%.3g ms\n",
		milliseconds);
#endif
#ifdef DEBUG
	int binidx = CalcGlobalIdx(0,0,2,numobins[0],numobins[1],
					numobins[2],binsperobinx,binsperobiny,binsperobinz);
	cout<<"binidx = "<<binidx;
	int *h_idxnupts;
	h_idxnupts = (int*)malloc(totalNUpts*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_idxnupts,d_idxnupts,totalNUpts*sizeof(int),
		cudaMemcpyDeviceToHost));
	int pts = 0;
	for(int b=0; b<numbins[0]*numbins[1]*numbins[1]; b++){
		if(h_binsize[b] > 0)
			cout <<"[debug ] Bin "<<b<<endl;
		for (int i=h_binstartpts[b]; i<h_binstartpts[b]+h_binsize[b]; i++){
			cout <<"[debug ] NUpts-index= "<< h_idxnupts[i]<<endl;
			pts++;
		}
	}
	cout<<"[debug ] totalpts = "<<pts<<endl;
	free(h_idxnupts);
	free(h_binstartpts);
	free(h_binsize);
#endif

	/* --------------------------------------------- */
	//        Determining Subproblem properties      //
	/* --------------------------------------------- */
	cudaEventRecord(start);
	n = numobins[0]*numobins[1]*numobins[2];
	cudaEventRecord(start);
	CalcSubProb_3d_v1<<<(n+1024-1)/1024, 1024>>>(binsperobinx, binsperobiny,
		binsperobinz, d_binsize, d_numsubprob, maxsubprobsize, numobins[0]*
		numobins[1]*numobins[2]);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcSubProb_3d_v1\t\t%.3g ms\n",
		milliseconds);
#endif
#ifdef DEBUG
	int* h_numsubprob;
	h_numsubprob = (int*) malloc(n*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_numsubprob,d_numsubprob,numobins[0]*numobins[1]*
		numobins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	for(int k=0; k<numobins[2]; k++){
		cout<<"[debug ] "<<endl;
		for(int j=0; j<numobins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numobins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<"s["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k
					<<"]= "<<setw(3)<<h_numsubprob[i+j*numobins[0]+k*
					numobins[1]*numobins[2]];
			}
			cout<<endl;
		}
	}
	free(h_numsubprob);
#endif
	cudaEventRecord(start);
	n = numobins[0]*numobins[1]*numobins[2];
	// Scanning a array with less length, so we don't need calculate temp_
	// storage_bytes here
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage,
		temp_storage_bytes, d_numsubprob, d_subprobstartpts+1, n));
	checkCudaErrors(cudaMemset(d_subprobstartpts,0,sizeof(int)));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tScan  numsubprob\t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	printf("[debug ] Subproblem start points\n");
	int* h_subprobstartpts;
	h_subprobstartpts = (int*) malloc((n+1)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_subprobstartpts,d_subprobstartpts,(numobins[0]*
		numobins[1]*numobins[2]+1)*sizeof(int),cudaMemcpyDeviceToHost));
	for(int k=0; k<numobins[2]; k++){
		if(k!=0)
			cout<<"[debug ] "<<endl;
		for(int j=0; j<numobins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numobins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<"s["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k
					<<"]= "<<setw(3)<<h_subprobstartpts[i+j*numobins[0]+k*
					numobins[1]*numobins[2]];
			}
			cout<<endl;
		}
	}
	printf("[debug ] Total number of subproblems (%d) = %d\n", n,
		h_subprobstartpts[n]);
	free(h_subprobstartpts);
	cout<<"[debug ] ---------------------------------------------------"<<endl;
#endif
	cudaEventRecord(start);
	int totalnumsubprob;
	checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[n],
		sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));
	MapBintoSubProb_3d_v1<<<(n+1024-1)/1024, 1024>>>(d_subprob_to_bin,
		d_subprobstartpts,d_numsubprob,n);
	assert(d_subprob_to_bin != NULL);
	d_plan->subprob_to_bin   = d_subprob_to_bin;
	d_plan->totalnumsubprob  = totalnumsubprob;
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Subproblem to Bin map\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	printf("[debug ] Map Subproblem to Bins\n");
	int* h_subprob_to_bin;
	h_subprob_to_bin   = (int*) malloc((totalnumsubprob)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_subprob_to_bin,d_subprob_to_bin,
		(totalnumsubprob)*sizeof(int),cudaMemcpyDeviceToHost));
	for(int j=0; j<totalnumsubprob; j++){
		cout<<"[debug ] ";
		cout <<"s["<<j<<"] = "<<setw(2)<<"b["<<h_subprob_to_bin[j]<<"]";
		cout<<endl;
	}
	free(h_subprob_to_bin);
#endif
	cudaFree(d_temp_storage);
	return 0;
}

int cuspread3d_blockgather(int nf1, int nf2, int nf3, int M, 
	cufinufft_plan *d_plan, int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=d_plan->spopts.nspread; 
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	FLT sigma=d_plan->spopts.upsampfac;
	int pirange=d_plan->spopts.pirange;
	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;

	int obin_size_x=d_plan->opts.gpu_obinsizex;
	int obin_size_y=d_plan->opts.gpu_obinsizey;
	int obin_size_z=d_plan->opts.gpu_obinsizez;
	int bin_size_x=d_plan->opts.gpu_binsizex;
	int bin_size_y=d_plan->opts.gpu_binsizey;
	int bin_size_z=d_plan->opts.gpu_binsizez;
	int numobins[3];
	numobins[0] = ceil((FLT) nf1/obin_size_x);
	numobins[1] = ceil((FLT) nf2/obin_size_y);
	numobins[2] = ceil((FLT) nf3/obin_size_z);

	int binsperobinx, binsperobiny, binsperobinz;
	binsperobinx = obin_size_x/bin_size_x+2;
	binsperobiny = obin_size_y/bin_size_y+2;
	binsperobinz = obin_size_z/bin_size_z+2;
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<obin_size_x<<"x"<<obin_size_y<<"x"<<obin_size_z<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numobins[0]<<"x"<<numobins[1]<<"x"<<
		numobins[2]<<"]"<<endl;
#endif

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	FLT* d_kz = d_plan->kz;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	int *d_binstartpts = d_plan->binstartpts;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;

	int totalnumsubprob=d_plan->totalnumsubprob;
	int *d_subprob_to_bin = d_plan->subprob_to_bin;

	cudaEventRecord(start);
	for(int t=0; t<blksize; t++){
		if(d_plan->opts.gpu_kerevalmeth == 1){
			size_t sharedplanorysize = obin_size_x*obin_size_y*obin_size_z
				*sizeof(CUCPX);
			if(sharedplanorysize > 49152){
				cout<<"error: not enough shared memory"<<endl;
				return 1;
			}
			Spread_3d_BlockGather_Horner<<<totalnumsubprob, 64, sharedplanorysize
				>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, M, ns,
					nf1, nf2, nf3, es_c, es_beta, sigma, d_binstartpts,
					obin_size_x, obin_size_y, obin_size_z,
					binsperobinx*binsperobiny*binsperobinz,d_subprob_to_bin,
					d_subprobstartpts, maxsubprobsize, numobins[0], numobins[1],
					numobins[2], d_idxnupts,pirange);
		}else{
			size_t sharedplanorysize = obin_size_x*obin_size_y*obin_size_z
				*sizeof(CUCPX);
			if(sharedplanorysize > 49152){
				cout<<"error: not enough shared memory"<<endl;
				return 1;
			}
			Spread_3d_BlockGather<<<totalnumsubprob, 64, sharedplanorysize>>>(
					d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, M, ns,
					nf1, nf2, nf3, es_c, es_beta, sigma, d_binstartpts,
					obin_size_x, obin_size_y, obin_size_z,
					binsperobinx*binsperobiny*binsperobinz,d_subprob_to_bin,
					d_subprobstartpts, maxsubprobsize, numobins[0], numobins[1],
					numobins[2], d_idxnupts,pirange);
		}
	}
#ifdef SPREADTIME
			float milliseconds = 0;
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("[time  ] \tKernel Spread_3d_BlockGather (%d)\t%.3g ms\n",
				milliseconds, d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}

int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M, 
	cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;
	int bin_size_x=d_plan->opts.gpu_binsizex;
	int bin_size_y=d_plan->opts.gpu_binsizey;
	int bin_size_z=d_plan->opts.gpu_binsizez;
	int numbins[3];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
	numbins[2] = ceil((FLT) nf2/bin_size_z);
#ifdef DEBUG
	cout<<"[debug ] Dividing the uniform grids to bin size["
		<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"x"<<d_plan->opts.gpu_binsizez<<"]"<<endl;
	cout<<"[debug ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"x"<<numbins[2]
		<<"]"<<endl;
#endif

	FLT*   d_kx = d_plan->kx;
	FLT*   d_ky = d_plan->ky;
	FLT*   d_kz = d_plan->kz;

#ifdef DEBUG
	FLT *h_kx;
	FLT *h_ky;
	FLT *h_kz;
	h_kx = (FLT*)malloc(M*sizeof(FLT));
	h_ky = (FLT*)malloc(M*sizeof(FLT));
	h_kz = (FLT*)malloc(M*sizeof(FLT));

	checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_ky,d_ky,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_kz,d_kz,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	for(int i=M-10; i<M; i++){
		cout<<"[debug ] ";
		cout <<"("<<setw(3)<<h_kx[i]<<","<<setw(3)<<h_ky[i]<<","<<setw(3)<<h_kz[i]
			<<")"<<endl;
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
	int pirange = d_plan->spopts.pirange;

	cudaEventRecord(start);
	checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*numbins[2]*
		sizeof(int)));
	CalcBinSize_noghost_3d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,nf3,bin_size_x,
		bin_size_y,bin_size_z,numbins[0],numbins[1],numbins[2],d_binsize,d_kx,
		d_ky,d_kz,d_sortidx,pirange);
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcBinSize_noghost_3d \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int *h_binsize;// For debug
	h_binsize     = (int*)malloc(numbins[0]*numbins[1]*numbins[2]*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*
		numbins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] bin size:"<<endl;
	for(int k=0; k<numbins[2]; k++){
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<" bin["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k<<"]="<<
					h_binsize[i+j*numbins[0]+k*numbins[0]*numbins[1]];
			}
			cout<<endl;
		}
	}
	free(h_binsize);
	cout<<"[debug ] ----------------------------------------------------"<<endl;
#endif
#ifdef DEBUG
	int *h_sortidx;
	h_sortidx = (int*)malloc(M*sizeof(int));

	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
		cudaMemcpyDeviceToHost));
	for(int i=0; i<10; i++){
		cout<<"[debug ] ";
		cout <<"point["<<setw(3)<<i<<"]="<<setw(3)<<h_sortidx[i]<<endl;
	}
#endif

	cudaEventRecord(start);
	int n=numbins[0]*numbins[1]*numbins[2];
	size_t temp_storage_bytes = 0;
	assert(d_temp_storage == NULL);
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes,
		d_binsize, d_binstartpts, n));
	// Allocate temporary storage for inclusive prefix scan
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes,
		d_binsize, d_binstartpts, n));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel BinStartPts_3d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int *h_binstartpts;
	h_binstartpts = (int*)malloc((numbins[0]*numbins[1]*numbins[2])*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]*
		numbins[1]*numbins[2])*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] Result of scan bin_size array:"<<endl;
	for(int k=0; k<numbins[2]; k++){
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<" bin["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k<<"]="<<
					h_binstartpts[i+j*numbins[0]+k*numbins[0]*numbins[1]];
			}
			cout<<endl;
		}
	}
	free(h_binstartpts);
	cout<<"[debug ] ---------------------------------------------------"<<endl;
#endif
	cudaEventRecord(start);
	CalcInvertofGlobalSortIdx_3d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
		bin_size_y,bin_size_z,numbins[0],numbins[1],numbins[2], d_binstartpts,
		d_sortidx,d_kx,d_ky,d_kz,d_idxnupts,pirange,nf1,nf2,nf3);
#ifdef DEBUG
	int *h_idxnupts;
	h_idxnupts = (int*)malloc(M*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_idxnupts,d_idxnupts,M*sizeof(int),
				cudaMemcpyDeviceToHost));
	for (int i=0; i<4; i++){
		cout <<"[debug ] idx="<< h_idxnupts[i]<<endl;
	}
	free(h_idxnupts);
#endif
	/* --------------------------------------------- */
	//        Determining Subproblem properties      //
	/* --------------------------------------------- */
	cudaEventRecord(start);
	CalcSubProb_3d_v2<<<(M+1024-1)/1024, 1024>>>(d_binsize,d_numsubprob,
			maxsubprobsize,numbins[0]*numbins[1]*numbins[2]);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcSubProb_3d_v2\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int* h_numsubprob;
	h_numsubprob = (int*) malloc(n*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_numsubprob,d_numsubprob,numbins[0]*numbins[1]*
				numbins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	for(int k=0; k<numbins[2]; k++){
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<" ns["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k<<"]="<<
					h_numsubprob[i+j*numbins[0]+k*numbins[0]*numbins[1]];
			}
			cout<<endl;
		}
	}
	free(h_numsubprob);
#endif
	// Scanning the same length array, so we don't need calculate
	// temp_storage_bytes here
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage,temp_storage_bytes,
		d_numsubprob, d_subprobstartpts+1, n));
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
	for(int k=0; k<numbins[2]; k++){
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<" ns["<<setw(1)<<i<<","<<setw(1)<<j<<","<<setw(1)<<k<<"]="<<
					h_subprobstartpts[i+j*numbins[0]+k*numbins[0]*numbins[1]];
			}
			cout<<endl;
		}
	}
	printf("[debug ] Total number of subproblems = %d\n", h_subprobstartpts[n]);
	free(h_subprobstartpts);
#endif
	int totalnumsubprob;
	checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[n],
		sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));
	MapBintoSubProb_3d_v2<<<(numbins[0]*numbins[1]+1024-1)/1024, 1024>>>(
		d_subprob_to_bin,d_subprobstartpts,d_numsubprob,numbins[0]*numbins[1]*
		numbins[2]);
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
	cout << totalnumsubprob << endl;
	for(int j=0; j<10; j++){
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

int cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan *d_plan,
	int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=d_plan->spopts.nspread;   // psi's support in terms of number of cells
	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;

	// assume that bin_size_x > ns/2;
	int bin_size_x=d_plan->opts.gpu_binsizex;
	int bin_size_y=d_plan->opts.gpu_binsizey;
	int bin_size_z=d_plan->opts.gpu_binsizez;
	int numbins[3];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
	numbins[2] = ceil((FLT) nf3/bin_size_z);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"x"<<d_plan->opts.gpu_binsizez<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
	cout<<ns<<endl;
#endif

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	FLT* d_kz = d_plan->kz;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;

	int totalnumsubprob=d_plan->totalnumsubprob;
	int *d_subprob_to_bin = d_plan->subprob_to_bin;

	FLT sigma=d_plan->spopts.upsampfac;
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	int pirange=d_plan->spopts.pirange;
	cudaEventRecord(start);
	size_t sharedplanorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*
			ceil(ns/2.0))*(bin_size_z+2*ceil(ns/2.0))*sizeof(CUCPX);
	if(sharedplanorysize > 49152){
		cout<<"error: not enough shared memory ("<<sharedplanorysize<<")"<<endl;
		return 1;
	}

	for(int t=0; t<blksize; t++){
		if(d_plan->opts.gpu_kerevalmeth){
			Spread_3d_Subprob_Horner<<<totalnumsubprob, 256,
				sharedplanorysize>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, 
				M, ns, nf1, nf2, nf3, sigma, d_binstartpts, d_binsize, bin_size_x,
				bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts,
				d_numsubprob, maxsubprobsize,numbins[0], numbins[1], numbins[2],
				d_idxnupts,pirange);
		}else{
			Spread_3d_Subprob<<<totalnumsubprob, 256,
				sharedplanorysize>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, 
				M, ns, nf1, nf2, nf3, es_c, es_beta, d_binstartpts, d_binsize, 
				bin_size_x,bin_size_y, bin_size_z, d_subprob_to_bin, 
				d_subprobstartpts,d_numsubprob, maxsubprobsize,numbins[0], 
				numbins[1], numbins[2],d_idxnupts,pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_3d_Subprob (%d) \t%.3g ms\n", milliseconds, 
		d_plan->opts.gpu_kerevalmeth);
#endif
	return 0;
}

#if 0
int cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M,
		const cufinufft_opts opts, cufinufft_plan *d_plan)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int maxsubprobsize=d_plan->opts.gpu_maxsubprobsize;
	int o_bin_size_x = d_plan->opts.gpu_obinsizex;
	int o_bin_size_y = d_plan->opts.gpu_obinsizey;
	int o_bin_size_z = d_plan->opts.gpu_obinsizez;
	int numobins[3];
	numobins[0] = ceil((FLT) nf1/o_bin_size_x);
	numobins[1] = ceil((FLT) nf2/o_bin_size_y);
	numobins[2] = ceil((FLT) nf3/o_bin_size_z);

	int bin_size_x=d_plan->opts.gpu_binsizex;
	int bin_size_y=d_plan->opts.gpu_binsizey;
	int bin_size_z=d_plan->opts.gpu_binsizez;
	int numbins[3];
	numbins[0] = numobins[0]*o_bin_size_x/bin_size_x;
	numbins[1] = numobins[1]*o_bin_size_y/bin_size_y;
	numbins[2] = numobins[2]*o_bin_size_z/bin_size_z;
#ifdef DEBUG
	cout<<"[debug ] Dividing the uniform grids to bin size["
		<<d_plan->opts.gpu_binsizex<<"x"<<d_plan->opts.gpu_binsizey<<"x"<<d_plan->opts.gpu_binsizez<<"]"<<endl;
	cout<<"[debug ] numbins = ["<<numobins[0]<<"x"<<numobins[1]<<"x"<<
		numobins[2]<<"]"<<endl;
#endif

	FLT*   d_kx = d_plan->kx;
	FLT*   d_ky = d_plan->ky;
	FLT*   d_kz = d_plan->kz;

#ifdef DEBUG
	FLT *h_kx, *h_ky, *h_kz;
	h_kx = (FLT*)malloc(M*sizeof(FLT));
	h_ky = (FLT*)malloc(M*sizeof(FLT));
	h_kz = (FLT*)malloc(M*sizeof(FLT));

	checkCudaErrors(cudaMemcpy(h_kx,d_kx,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_ky,d_ky,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_kz,d_kz,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		cout<<"[debug ] ";
		cout <<"("<<setw(3)<<h_kx[i]<<","<<setw(3)<<h_ky[i]<<","<<h_kz[i]<<")"
			<<endl;
	}
#endif
	int *d_binsize = d_plan->binsize;
	int *d_sortidx = d_plan->sortidx;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_numnupts = d_plan->numnupts;
	void*d_temp_storage = NULL;
	int *d_idxnupts = d_plan->idxnupts;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_subprob_to_bin = NULL;
	int *d_subprob_to_nupts = NULL;

	cudaEventRecord(start);
	checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*numbins[2]*
		sizeof(int)));
	LocateNUptstoBins<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,nf3,bin_size_x,
		bin_size_y,bin_size_z,numbins[0],numbins[1],numbins[2],d_binsize,d_kx,
		d_ky,d_kz,d_sortidx);
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcBinSize_noghost_3d \t\t%.3g ms\n",
		milliseconds);
#endif
#ifdef DEBUG
	int *h_binsize;// For debug
	h_binsize     = (int*)malloc(numbins[0]*numbins[1]*numbins[2]*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*
		numbins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] bin size:"<<endl;
	for(int k=0; k<numbins[2]; k++){
		cout<<"[debug ] "<<endl;
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout<<" b["<<setw(3)<<i<<","<<setw(3)<<j<<","<<setw(3)<<k
					<<"]="<<h_binsize[i+j*numbins[0]+k*numbins[0]*numbins[1]];
			}
			cout<<endl;
		}
	}
	free(h_binsize);
	cout<<"[debug ] ---------------------------------------------------"<<endl;
	int *h_sortidx;
	h_sortidx = (int*)malloc(M*sizeof(int));

	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
		cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		cout <<"[debug ] point["<<setw(3)<<i<<"]="<<setw(3)<<h_sortidx[i]<<endl;
	}

#endif
	int n=numbins[0]*numbins[1]*numbins[2];
	size_t temp_storage_bytes = 0;
	assert(d_temp_storage == NULL);
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,
		temp_storage_bytes, d_binsize, d_binstartpts, n));
	// Allocate temporary storage for inclusive prefix scan
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage,
		temp_storage_bytes, d_binsize, d_binstartpts, n));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel BinStartPts_3d \t\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	int *h_binstartpts;
	h_binstartpts = (int*)malloc((numbins[0]*numbins[1]*numbins[2])*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]*
		numbins[1]*numbins[2])*sizeof(int),cudaMemcpyDeviceToHost));
	cout<<"[debug ] Result of scan bin_size array:"<<endl;
	for(int k=0; k<numbins[2]; k++){
		cout<<"[debug ] "<<endl;
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout<<" b["<<setw(3)<<i<<","<<setw(3)<<j<<","<<setw(3)<<k
					<<"]="<<h_binstartpts[i+j*numbins[0]+k*numbins[0]*
					numbins[1]];
			}
			cout<<endl;
		}
	}
	free(h_binstartpts);
	cout<<"[debug ] ----------------------------------------------------"<<endl;
#endif
	cudaEventRecord(start);
	CalcInvertofGlobalSortIdx_3d<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
		bin_size_y,bin_size_z,numbins[0],numbins[1],numbins[2],d_binstartpts,
		d_sortidx,d_kx,d_ky,d_kz,d_idxnupts);
#ifdef DEBUG
	int *h_idxnupts;
	h_idxnupts = (int*)malloc(M*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_idxnupts,d_idxnupts,M*sizeof(int),
		cudaMemcpyDeviceToHost));
	for (int i=0; i<M; i++){
		cout <<"[debug ] NUpts-index= "<< h_idxnupts[i]<<endl;
	}
	free(h_idxnupts);
#endif
	/* --------------------------------------------- */
	//        Determining Subproblem properties      //
	/* --------------------------------------------- */
#if 0
	cudaEventRecord(start);
	CalcSubProb_3d<<<(M+1024-1)/1024, 1024>>>(d_binsize, d_numsubprob,
		maxsubprobsize, numbins[0]*numbins[1]*numbins[2]);
#ifdef DEBUG
	int* h_numsubprob;
	h_numsubprob = (int*) malloc(n*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_numsubprob,d_numsubprob,numbins[0]*numbins[1]*
		numbins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	for(int k=0; k<numbins[2]; k++){
		cout<<"[debug ] "<<endl;
		for(int j=0; j<numbins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numbins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<"nsub["<<setw(3)<<i<<","<<setw(3)<<j<<","<<setw(3)<<k
					<<"] = "<<setw(2)<<h_numsubprob[i+j*numbins[0]+k*
					numbins[1]*numbins[2]];
			}
			cout<<endl;
		}
	}
	free(h_numsubprob);
#endif
#endif
	cudaEventRecord(start);
	threadsPerBlock.x=8;
	threadsPerBlock.y=8;
	threadsPerBlock.z=8;
	blocks.x = (threadsPerBlock.x+numobins[0]-1)/threadsPerBlock.x;
	blocks.y = (threadsPerBlock.y+numobins[1]-1)/threadsPerBlock.y;
	blocks.z = (threadsPerBlock.z+numobins[2]-1)/threadsPerBlock.z;
	CalcSubProb_3d<<<blocks, threadsPerBlock>>>(bin_size_x, bin_size_y,
		bin_size_z, o_bin_size_x, o_bin_size_y, o_bin_size_z, numbins[0],
		numbins[1], numbins[2], numobins[0], numobins[1], numobins[2],
		d_binsize,d_numsubprob, d_numnupts, maxsubprobsize);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel CalcSubProb_3d\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	cout<<"[debug ] ---------------------------------------------------"<<endl;
	cout << "[debug ] Number of nupts effecting the output bins:" << endl;
	n = numobins[0]*numobins[1]*numobins[2];
	int* h_numnupts;
	h_numnupts = (int*) malloc(n*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_numnupts,d_numnupts,numobins[0]*numobins[1]*
		numobins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	for(int k=0; k<numobins[2]; k++){
		cout<<"[debug ] "<<endl;
		for(int j=0; j<numobins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numobins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<"s["<<setw(3)<<i<<","<<setw(3)<<j<<","<<setw(3)<<k
					<<"] = "<<setw(2)<<h_numnupts[i+j*numobins[0]+k*
					numobins[1]*numobins[2]];
			}
			cout<<endl;
		}
	}
	cout<<"[debug ] ---------------------------------------------------"<<endl;
	free(h_numnupts);
#endif
#ifdef DEBUG
	int* h_numsubprob;
	h_numsubprob = (int*) malloc(n*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_numsubprob,d_numsubprob,numobins[0]*numobins[1]*
		numobins[2]*sizeof(int),cudaMemcpyDeviceToHost));
	for(int k=0; k<numobins[2]; k++){
		cout<<"[debug ] "<<endl;
		for(int j=0; j<numobins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numobins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<"s["<<setw(3)<<i<<","<<setw(3)<<j<<","<<setw(3)<<k
					<<"] = "<<setw(2)<<h_numsubprob[i+j*numobins[0]+k*
					numobins[1]*numobins[2]];
			}
			cout<<endl;
		}
	}
	cout<<"[debug ] ---------------------------------------------------"<<endl;
	free(h_numsubprob);
#endif
	n = numobins[0]*numobins[1]*numobins[2];
	// Scanning a array with less length, so we don't need calculate temp_
	// storage_bytes here
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage,
		temp_storage_bytes, d_numsubprob, d_subprobstartpts+1, n));
	checkCudaErrors(cudaMemset(d_subprobstartpts,0,sizeof(int)));
#ifdef DEBUG
	printf("[debug ] Subproblem start points\n");
	int* h_subprobstartpts;
	h_subprobstartpts = (int*) malloc((n+1)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_subprobstartpts,d_subprobstartpts,(numobins[0]*
		numobins[1]*numobins[2]+1)*sizeof(int),cudaMemcpyDeviceToHost));
	for(int k=0; k<numobins[2]; k++){
		if(k!=0)
			cout<<"[debug ] "<<endl;
		for(int j=0; j<numobins[1]; j++){
			cout<<"[debug ] ";
			for(int i=0; i<numobins[0]; i++){
				if(i!=0) cout<<" ";
				cout <<"s["<<setw(3)<<i<<","<<setw(3)<<j<<","<<setw(3)<<k
					<<"] = "<<setw(2)<<h_subprobstartpts[i+j*numobins[0]+k*
					numobins[1]*numobins[2]];
			}
			cout<<endl;
		}
	}
	printf("[debug ] Total number of subproblems %d= %d\n", n,
		h_subprobstartpts[n]);
	free(h_subprobstartpts);
	cout<<"[debug ] ---------------------------------------------------"<<endl;
#endif
	cudaEventRecord(start);
	int totalnumsubprob;
	checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[n],
		sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_subprob_to_nupts,totalnumsubprob*sizeof(int)));
	threadsPerBlock.x=8;
	threadsPerBlock.y=8;
	threadsPerBlock.z=8;
	blocks.x = (threadsPerBlock.x+numobins[0]-1)/threadsPerBlock.x;
	blocks.y = (threadsPerBlock.y+numobins[1]-1)/threadsPerBlock.y;
	blocks.z = (threadsPerBlock.z+numobins[2]-1)/threadsPerBlock.z;

	MapBintoSubProb_3d<<<blocks, threadsPerBlock>>>(d_subprobstartpts,
		d_subprob_to_bin, d_subprob_to_nupts, bin_size_x, bin_size_y,
		bin_size_z, o_bin_size_x, o_bin_size_y, o_bin_size_z, numbins[0],
		numbins[1], numbins[2], numobins[0], numobins[1], numobins[2],
		d_binsize,d_numsubprob, d_numnupts, maxsubprobsize);

	assert(d_subprob_to_bin != NULL);
	assert(d_subprob_to_nupts != NULL);
	d_plan->subprob_to_bin   = d_subprob_to_bin;
	d_plan->subprob_to_nupts = d_subprob_to_nupts;
	d_plan->totalnumsubprob  = totalnumsubprob;
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Subproblem to Bin map\t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	printf("[debug ] Map Subproblem to Bins\n");
	int* h_subprob_to_bin, *h_subprob_to_nupts;
	h_subprob_to_bin   = (int*) malloc((totalnumsubprob)*sizeof(int));
	h_subprob_to_nupts = (int*) malloc((totalnumsubprob)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_subprob_to_bin,d_subprob_to_bin,
		(totalnumsubprob)*sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_subprob_to_nupts,d_subprob_to_nupts,
		(totalnumsubprob)*sizeof(int),cudaMemcpyDeviceToHost));
	for(int j=0; j<totalnumsubprob; j++){
		cout<<"[debug ] ";
		cout <<"s["<<j<<"] = "<<setw(2)<<"b["<<h_subprob_to_bin[j]<<"]"<<
			", nupts = "<<h_subprob_to_nupts[j];
		cout<<endl;
	}
	free(h_subprob_to_bin);
#endif
	cudaFree(d_temp_storage);
	return 0;
}
#endif

