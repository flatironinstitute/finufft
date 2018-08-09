#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

// try another library cub
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuComplex.h>
#include "spread.h"

using namespace std;

int cnufft_allocgpumemory(int ms, int mt, int nf1, int nf2, int M, int* fw_width, spread_opts opts, spread_devicemem *d_mem)
{
	d_mem->byte_now=0;
	// No extra memory is needed in idriven method;
	switch(opts.method)
	{
		case 2:
			{
				//int total_mem_in_bytes=
				checkCudaErrors(cudaMalloc(&d_mem->kxsorted,M*sizeof(FLT)));
				checkCudaErrors(cudaMalloc(&d_mem->kysorted,M*sizeof(FLT)));
				checkCudaErrors(cudaMalloc(&d_mem->csorted,M*sizeof(gpuComplex)));
				checkCudaErrors(cudaMalloc(&d_mem->sortidx,M*sizeof(int)));

				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				checkCudaErrors(cudaMalloc(&d_mem->binsize,numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_mem->binstartpts,(numbins[0]*numbins[1])*sizeof(int)));
			}
			break;
		case 4:
			{
				checkCudaErrors(cudaMalloc(&d_mem->kxsorted,M*sizeof(FLT)));
				checkCudaErrors(cudaMalloc(&d_mem->kysorted,M*sizeof(FLT)));
				checkCudaErrors(cudaMalloc(&d_mem->csorted,M*sizeof(gpuComplex)));

				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				checkCudaErrors(cudaMalloc(&d_mem->binsize,numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_mem->sortidx,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_mem->binstartpts,(numbins[0]*numbins[1]+1)*sizeof(int)));
			}
			break;
		case 5:
			{
				int numbins[2];
				numbins[0] = ceil((FLT) nf1/opts.bin_size_x);
				numbins[1] = ceil((FLT) nf2/opts.bin_size_y);
				checkCudaErrors(cudaMalloc(&d_mem->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_mem->sortidx,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_mem->numsubprob,  numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_mem->binsize,     numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_mem->binstartpts, numbins[0]*numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_mem->subprobstartpts,(numbins[0]*numbins[1]+1)*sizeof(int)));
			}
			break;
	}
	checkCudaErrors(cudaMalloc(&d_mem->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_mem->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_mem->c,M*sizeof(gpuComplex)));

	size_t pitch;
	checkCudaErrors(cudaMallocPitch((void**) &d_mem->fw, &pitch,nf1*sizeof(gpuComplex),nf2));
	*fw_width = pitch/sizeof(gpuComplex);

	checkCudaErrors(cudaMalloc(&d_mem->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_mem->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_mem->fk,ms*mt*sizeof(gpuComplex)));

	return 0;
}

int cnufft_copycpumem_to_gpumem(int M, FLT *h_kx, FLT* h_ky, CPX *h_c, int nf1, int nf2, FLT* h_fwkerhalf1, FLT* h_fwkerhalf2, 
                                spread_devicemem *d_mem)
{
	checkCudaErrors(cudaMemcpy(d_mem->kx,h_kx,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mem->ky,h_ky,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mem->c, h_c,M*sizeof(gpuComplex),cudaMemcpyHostToDevice));

	if(h_fwkerhalf1 != NULL)
		checkCudaErrors(cudaMemcpy(d_mem->fwkerhalf1,h_fwkerhalf1,(nf1/2+1)*sizeof(FLT),cudaMemcpyHostToDevice));
	if(h_fwkerhalf2 != NULL)
		checkCudaErrors(cudaMemcpy(d_mem->fwkerhalf2,h_fwkerhalf2,(nf2/2+1)*sizeof(FLT),cudaMemcpyHostToDevice));
	
	return 0;
}

int cnufft_copygpumem_to_cpumem_fw(int nf1, int nf2, int fw_width, CPX* h_fw, spread_devicemem *d_mem)
{
	checkCudaErrors(cudaMemcpy2D(h_fw,nf1*sizeof(gpuComplex),d_mem->fw,fw_width*sizeof(gpuComplex),
				nf1*sizeof(gpuComplex),nf2,cudaMemcpyDeviceToHost));

	return 0;
}

void cnufft_free_gpumemory(spread_opts opts, spread_devicemem *d_mem)
{
	cudaFree(d_mem->fw);
	cudaFree(d_mem->kx);
	cudaFree(d_mem->ky);
	cudaFree(d_mem->c);
	cudaFree(d_mem->fwkerhalf1);
	cudaFree(d_mem->fwkerhalf2);
	switch(opts.method)
	{
		case 2:
			{
				checkCudaErrors(cudaFree(d_mem->kxsorted));
				checkCudaErrors(cudaFree(d_mem->kysorted));
				checkCudaErrors(cudaFree(d_mem->csorted));
				checkCudaErrors(cudaFree(d_mem->sortidx));
				checkCudaErrors(cudaFree(d_mem->binsize));
				checkCudaErrors(cudaFree(d_mem->binstartpts));
				checkCudaErrors(cudaFree(d_mem->temp_storage));
			}
			break;
		case 4:
			{
				checkCudaErrors(cudaFree(d_mem->kxsorted));
				checkCudaErrors(cudaFree(d_mem->kysorted));
				checkCudaErrors(cudaFree(d_mem->csorted));
				checkCudaErrors(cudaFree(d_mem->binsize));
				checkCudaErrors(cudaFree(d_mem->sortidx));
				checkCudaErrors(cudaFree(d_mem->binstartpts));
				checkCudaErrors(cudaFree(d_mem->temp_storage));
			}
			break;
		case 5:
			{
				checkCudaErrors(cudaFree(d_mem->idxnupts));
				checkCudaErrors(cudaFree(d_mem->sortidx));
				checkCudaErrors(cudaFree(d_mem->numsubprob));
				checkCudaErrors(cudaFree(d_mem->binsize));
				checkCudaErrors(cudaFree(d_mem->binstartpts));
				checkCudaErrors(cudaFree(d_mem->subprobstartpts));
				checkCudaErrors(cudaFree(d_mem->temp_storage));
				checkCudaErrors(cudaFree(d_mem->subprob_to_bin));
			}
			break;
	}
}

int cnufftspread2d_gpu(int ms, int mt, int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX *h_c, spread_opts opts, spread_devicemem* d_mem)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	int fw_width;

	if(opts.pirange){
		for(int i=0; i<M; i++){
			h_kx[i]=RESCALE(h_kx[i], nf1, opts.pirange);
			h_ky[i]=RESCALE(h_ky[i], nf2, opts.pirange);
		}
	}
	cudaEventRecord(start);
	ier = cnufft_allocgpumemory(ms, mt, nf1, nf2, M, &fw_width, opts, d_mem);
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Allocating GPU memory " << milliseconds <<" ms"<<endl;
#endif
	cudaEventRecord(start);
	ier = cnufft_copycpumem_to_gpumem(M, h_kx, h_ky, h_c, nf1, nf2, NULL, NULL, d_mem);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Copying memory from host to device " << milliseconds <<" s"<<endl;
#endif

	switch(opts.method)
	{
		case 1:
			{
				cudaEventRecord(start);
				ier = cnufftspread2d_gpu_idriven(nf1, nf2, fw_width, M, opts, d_mem);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_idriven"<<endl;
					return 0;
				}
			}
			break;
		case 2:
			{
				cudaEventRecord(start);
				ier = cnufftspread2d_gpu_idriven_sorted(nf1, nf2, fw_width, M, opts, d_mem);
			}
			break;
		case 4:
			{
				cudaEventRecord(start);
				ier = cnufftspread2d_gpu_hybrid(nf1, nf2, fw_width, M, opts, d_mem);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_hybrid"<<endl;
					return 0;
				}
			}
			break;
		case 5:
			{
				cudaEventRecord(start);
				ier = cnufftspread2d_gpu_subprob(nf1, nf2, fw_width, M, opts, d_mem);
				if(ier != 0 ){
					cout<<"error: cnufftspread2d_gpu_hybrid"<<endl;
					return 0;
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1,2,4 or 5"<<endl;
			return 0;
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Spread " << milliseconds <<" ms"<<endl;
#endif
	cudaEventRecord(start);
	ier = cnufft_copygpumem_to_cpumem_fw(nf1, nf2, fw_width, h_fw, d_mem);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Copying memory from device to host " << milliseconds <<" ms"<<endl;
#endif
	cnufft_free_gpumemory(opts, d_mem);

	return ier;
}

int cnufftspread2d_gpu_simple(int nf1, int nf2, int fw_width, gpuComplex* d_fw, int M, FLT *d_kx,
		FLT *d_ky, gpuComplex *d_c, spread_opts opts, int binx, int biny)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;
	int bin_size_x=opts.bin_size_x;
	int bin_size_y=opts.bin_size_y;

	// assume that bin_size_x > ns/2;
	cudaEventRecord(start);
	threadsPerBlock.x = opts.nthread_x;
	threadsPerBlock.y = opts.nthread_y;
	blocks.x = 1;
	blocks.y = 1;
	size_t sharedmemorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*sizeof(gpuComplex);
	if(sharedmemorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}
	// blockSize must be a multiple of bin_size_x
	Spread_2d_Simple<<<blocks, threadsPerBlock, sharedmemorysize>>>(d_kx, d_ky, d_c, 
			d_fw, M, ns, nf1, nf2, 
			es_c, es_beta, fw_width, 
			M, bin_size_x, bin_size_y, 
			binx, biny);
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_2d_Simple \t\t%.3g ms\n", milliseconds);
#endif
	return 0;
}

int cnufftspread2d_gpu_idriven(int nf1, int nf2, int fw_width, int M, spread_opts opts, spread_devicemem *d_mem)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;

	FLT* d_kx = d_mem->kx;
	FLT* d_ky = d_mem->ky;
	gpuComplex* d_c = d_mem->c;
	gpuComplex* d_fw = d_mem->fw;

	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	cudaEventRecord(start);
	if(opts.Horner){
		Spread_2d_Idriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c, d_fw, M, ns,
				nf1, nf2, es_c, es_beta, fw_width);
	}else{
		Spread_2d_Idriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c, d_fw, M, ns,
				nf1, nf2, es_c, es_beta, fw_width);
	}

#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_2d_Idriven \t%.3g ms\n", milliseconds);
#endif
	return 0;
}

int cnufftspread2d_gpu_idriven_sorted(int nf1, int nf2, int fw_width, int M, spread_opts opts, spread_devicemem *d_mem)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;

	int bin_size_x=opts.bin_size_x;
	int bin_size_y=opts.bin_size_y;
	int numbins[2];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);

	FLT* d_kx = d_mem->kx;
	FLT* d_ky = d_mem->ky;
	gpuComplex* d_c = d_mem->c;
	gpuComplex* d_fw = d_mem->fw;

	FLT *d_kxsorted = d_mem->kxsorted;
	FLT *d_kysorted = d_mem->kysorted;
	gpuComplex *d_csorted = d_mem->csorted;

	int *d_binsize = d_mem->binsize;
	int *d_binstartpts = d_mem->binstartpts;
	int *d_sortidx = d_mem->sortidx;
	d_mem->temp_storage = NULL;
	void*d_temp_storage = d_mem->temp_storage;

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
	cudaEventRecord(start);
	int n=numbins[0]*numbins[1];
	size_t temp_storage_bytes = 0;
	CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_binsize, d_binstartpts, n));
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_binsize, d_binstartpts+1, n));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel BinStartPts_2d \t\t\t%.3g ms\n", milliseconds);
#endif
	cudaEventRecord(start);
	PtsRearrage_noghost_2d<<<(M+1024-1)/1024,1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0],
			numbins[1], d_binstartpts, d_sortidx, d_kx, d_kxsorted,
			d_ky, d_kysorted, d_c, d_csorted);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel PtsRearrange_noghost_2d \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_kx,d_kxsorted,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_ky,d_kysorted,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_c,d_csorted,M*sizeof(gpuComplex),cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		printf("sortidx = %d, (x,y) = (%.3g, %.3g), c=(%f, %f)\n", h_sortidx[i], h_kx[i], h_ky[i], h_c[i].real(), h_c[i].imag());
	}
#endif 
	cudaEventRecord(start);
	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	Spread_2d_Idriven<<<blocks, threadsPerBlock>>>(d_kxsorted, d_kysorted, d_csorted, d_fw, M, ns,
			nf1, nf2, es_c, es_beta, fw_width);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_2d_Idriven \t\t%.3g ms\n", milliseconds);
#endif
	return 0;
}

int cnufftspread2d_gpu_hybrid(int nf1, int nf2, int fw_width, int M, spread_opts opts, spread_devicemem *d_mem)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;

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

	FLT* d_kx = d_mem->kx;
	FLT* d_ky = d_mem->ky;
	gpuComplex* d_c = d_mem->c;
	gpuComplex* d_fw = d_mem->fw;

	int *d_binsize = d_mem->binsize;
	int *d_binstartpts = d_mem->binstartpts;
	int *d_sortidx = d_mem->sortidx;

	// assume that bin_size_x > ns/2;
	FLT *d_kxsorted = d_mem->kxsorted;
	FLT *d_kysorted = d_mem->kysorted;
	gpuComplex *d_csorted = d_mem->csorted;
	d_mem->temp_storage = NULL;
	void *d_temp_storage = d_mem->temp_storage;

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
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_binsize, d_binstartpts+1, n));
	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes)); // Allocate temporary storage for inclusive prefix scan
	CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_binsize, d_binstartpts+1, n));
	checkCudaErrors(cudaMemset(d_binstartpts,0,sizeof(int)));
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel BinStartPts_2d \t\t\t%.3g ms\n", milliseconds);
#endif

#ifdef DEBUG
	int *h_binstartpts;
	h_binstartpts = (int*)malloc((numbins[0]*numbins[1]+1)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]*numbins[1]+1)*sizeof(int),
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
	cout<<"[debug ] Total number of nonuniform pts (include those in ghost bins) = "
		<< setw(4)<<h_binstartpts[numbins[0]*numbins[1]]<<endl;
	free(h_binstartpts);
	cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

	cudaEventRecord(start);
	PtsRearrage_noghost_2d<<<(M+1024-1)/1024,1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0],
			numbins[1], d_binstartpts, d_sortidx, d_kx, d_kxsorted,
			d_ky, d_kysorted, d_c, d_csorted);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel PtsRearrange_noghost_2d \t\t%.3g ms\n", milliseconds);
#endif
#ifdef DEBUG
	FLT *h_kxsorted, *h_kysorted;
	CPX *h_csorted;
	h_kxsorted = (FLT*)malloc(M*sizeof(FLT));
	h_kysorted = (FLT*)malloc(M*sizeof(FLT));
	h_csorted  = (CPX*)malloc(M*sizeof(CPX));
	checkCudaErrors(cudaMemcpy(h_kxsorted,d_kxsorted,M*sizeof(FLT),
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_kysorted,d_kysorted,M*sizeof(FLT),
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_csorted,d_csorted,M*sizeof(CPX),
				cudaMemcpyDeviceToHost));
	for (int i=0; i<10; i++){
		cout <<"[debug ] (x,y) = ("<<setw(10)<<h_kxsorted[i]<<","
			<<setw(10)<<h_kysorted[i]<<"), bin# =  "
			<<(floor(h_kxsorted[i]/bin_size_x))+numbins[0]*(floor(h_kysorted[i]/bin_size_y))<<endl;
	}
	free(h_kysorted);
	free(h_kxsorted);
	free(h_csorted);
#endif

	cudaEventRecord(start);
	threadsPerBlock.x = 16;
	threadsPerBlock.y = 16;
	blocks.x = numbins[0];
	blocks.y = numbins[1];
	size_t sharedmemorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*sizeof(gpuComplex);
	if(sharedmemorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}
	// blockSize must be a multiple of bin_size_x
	Spread_2d_Hybrid<<<blocks, threadsPerBlock, sharedmemorysize>>>(d_kxsorted, d_kysorted, d_csorted, 
			d_fw, M, ns, nf1, nf2, 
			es_c, es_beta, fw_width, 
			d_binstartpts, d_binsize, 
			bin_size_x, bin_size_y);
#ifdef SPREADTIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Spread_2d_Hybrid \t\t%.3g ms\n", milliseconds);
#endif
	return 0;
}

int cnufftspread2d_gpu_subprob(int nf1, int nf2, int fw_width, int M, spread_opts opts, spread_devicemem *d_mem)
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

	FLT* d_kx = d_mem->kx;
	FLT* d_ky = d_mem->ky;
	gpuComplex* d_c = d_mem->c;
	gpuComplex* d_fw = d_mem->fw;

	int *d_binsize = d_mem->binsize;
	int *d_binstartpts = d_mem->binstartpts;
	int *d_sortidx = d_mem->sortidx;
	int *d_numsubprob = d_mem->numsubprob;
	int *d_subprobstartpts = d_mem->subprobstartpts;
	int *d_idxnupts = d_mem->idxnupts;
	d_mem->subprob_to_bin = NULL;
	int *d_subprob_to_bin = d_mem->subprob_to_bin;
	d_mem->temp_storage = NULL;
	void *d_temp_storage = d_mem->temp_storage;

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
	size_t sharedmemorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*sizeof(gpuComplex);
	if(sharedmemorysize > 49152){
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	Spread_2d_Subprob<<<totalnumsubprob, 256, sharedmemorysize>>>(d_kx, d_ky, d_c,
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
	printf("[time  ] \tKernel Spread_2d_Subprob_V2 \t\t%.3g ms\n", milliseconds);
#endif
	return 0;
}
