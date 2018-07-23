#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
// idriven coarse grained
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>

// try another library cub
#include <cub/device/device_radix_sort.cuh>

#include <cuComplex.h>
#include "spread.h"

using namespace std;

int cnufftspread2d_gpu(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX *h_c, spread_opts opts)
{
#if 1
	if(opts.pirange){
		for(int i=0; i<M; i++){
			h_kx[i]=RESCALE(h_kx[i], nf1, opts.pirange);
			h_ky[i]=RESCALE(h_ky[i], nf2, opts.pirange);
		}
	}
#endif
	return cnufftspread2d_gpu_idriven(nf1, nf2, h_fw, M, h_kx, h_ky, h_c, opts);
}

int cnufftspread2d_gpu_odriven(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX *h_c, spread_opts opts)
{
	checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	CNTime timer;
	FLT k_spread_time=0.0;
	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;

	FLT *d_kx,*d_ky;
	gpuComplex *d_c,*d_fw;
	// Parameter setting
	int numbins[2];
	int totalnupts;
	int nbin_block_x, nbin_block_y;
	int bin_size_x=opts.bin_size_x;
	int bin_size_y=opts.bin_size_y;

	int *d_binsize;
	int *d_binstartpts;
	int *d_sortidx;

	numbins[0] = ceil(nf1/bin_size_x)+2;
	numbins[1] = ceil(nf2/bin_size_y)+2;
	// assume that bin_size_x > ns/2;
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<opts.bin_size_x<<"x"<<opts.bin_size_y<<"]"<<endl;
	cout<<"[info  ] numbins (including ghost bins) = ["
		<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif
	FLT *d_kxsorted,*d_kysorted;
	gpuComplex *d_csorted;
	int *h_binsize, *h_binstartpts, *h_sortidx; // For debug

	timer.restart();
	checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(gpuComplex)));
	//checkCudaErrors(cudaMalloc(&d_fw,nf1*nf2*sizeof(gpuComplex)));
	int fw_width;
	size_t pitch;
	checkCudaErrors(cudaMallocPitch((void**) &d_fw, &pitch,nf1*sizeof(gpuComplex),nf2));
	fw_width = pitch/sizeof(gpuComplex);

	checkCudaErrors(cudaMalloc(&d_binsize,numbins[0]*numbins[1]*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_sortidx,M*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_binstartpts,(numbins[0]*numbins[1]+1)*sizeof(int)));
#ifdef TIME
	cout<<"[time  ]"<< " Allocating GPU memory " << timer.elapsedsec() <<" s"<<endl;
#endif

	timer.restart();
	checkCudaErrors(cudaMemcpy(d_kx,h_kx,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ky,h_ky,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c,h_c,M*sizeof(gpuComplex),cudaMemcpyHostToDevice));
#ifdef TIME
	cout<<"[time  ]"<< " Copying memory from host to device " << timer.elapsedsec() <<" s"<<endl;
#endif

	h_binsize     = (int*)malloc(numbins[0]*numbins[1]*sizeof(int));
	h_sortidx     = (int*)malloc(M*sizeof(int));
	h_binstartpts = (int*)malloc((numbins[0]*numbins[1]+1)*sizeof(int));
	checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*sizeof(int)));
	timer.restart();
	CalcBinSize_2d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,bin_size_x,bin_size_y,
			numbins[0],numbins[1],d_binsize,
			d_kx,d_ky,d_sortidx);
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " Kernel CalcBinSize_2d (#blocks, #threads)=("<<(M+1024-1)/1024<<","<<1024<<") takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*sizeof(int),
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
				cudaMemcpyDeviceToHost));
	cout<<"[debug ] Before fill in the ghost bin size:"<<endl;
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<" bin["<<setw(3)<<i<<","<<setw(3)<<j<<"]="<<h_binsize[i+j*numbins[0]];
		}
		cout<<endl;
	}
	cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif
	timer.restart();
	threadsPerBlock.x = 32;
	threadsPerBlock.y = 32;// doesn't work for 64, doesn't know why
	if(threadsPerBlock.x*threadsPerBlock.y < 1024){
		cout<<"error: number of threads in a block exceeds max num 1024("
			<<threadsPerBlock.x*threadsPerBlock.y<<")"<<endl;
		return 1;
	}
	blocks.x = (numbins[0]+threadsPerBlock.x-1)/threadsPerBlock.x;
	blocks.y = (numbins[1]+threadsPerBlock.y-1)/threadsPerBlock.y;
	FillGhostBin_2d<<<blocks,threadsPerBlock>>>(numbins[0],numbins[1],d_binsize);
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " Kernel FillGhostBin_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*sizeof(int),
				cudaMemcpyDeviceToHost));
	cout<<"[debug ] After fill in the ghost bin size:"<<endl;
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<"bin["<<setw(3)<<i<<","<<setw(3)<<j<<"] = "<<h_binsize[i+j*numbins[0]];
		}
		cout<<endl;
	}
	cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

	timer.restart();
	int n=numbins[0]*numbins[1];
	int scanblocksize=1024;
	int numscanblocks=ceil((double)n/scanblocksize);
	int* d_scanblocksum, *d_scanblockstartpts;
#ifdef DEBUG
	printf("[debug ] n=%d, numscanblocks=%d\n",n,numscanblocks);
#endif 
	checkCudaErrors(cudaMalloc(&d_scanblocksum,numscanblocks*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_scanblockstartpts,(numscanblocks+1)*sizeof(int)));

	for(int i=0;i<numscanblocks;i++){
		int nelemtoscan=(n-scanblocksize*i)>scanblocksize ? scanblocksize : n-scanblocksize*i;
		prescan<<<1, scanblocksize/2>>>(nelemtoscan,d_binsize+i*scanblocksize,
				d_binstartpts+i*scanblocksize,d_scanblocksum+i);
	}
#ifdef DEBUG
	int* h_scanblocksum;
	h_scanblocksum     =(int*) malloc(numscanblocks*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_scanblocksum,d_scanblocksum,numscanblocks*sizeof(int),
				cudaMemcpyDeviceToHost));
	for(int i=0;i<numscanblocks;i++){
		cout<<"[debug ] scanblocksum["<<i<<"]="<<h_scanblocksum[i]<<endl;
	}
#endif
	int next = pow(2, ceil(log(numscanblocks+1)/log(2)));
	if(next > 2048){
		cout<<"error: number of elements to sort exceed the prescan capability"<<endl;
		return 1;
	}
	prescan<<<1, next/2>>>(numscanblocks,d_scanblocksum,d_scanblockstartpts,d_scanblockstartpts+numscanblocks);
#ifdef DEBUG
	int* h_scanblockstartpts = (int*) malloc((numscanblocks+1)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_scanblockstartpts,d_scanblockstartpts,(numscanblocks+1)*sizeof(int),
				cudaMemcpyDeviceToHost));
	for(int i=0;i<numscanblocks+1;i++){
		cout<<"[debug ] scanblockstartpts["<<i<<"]="<<h_scanblockstartpts[i]<<endl;
	}
#endif
	uniformUpdate<<<numscanblocks,scanblocksize>>>(n,d_binstartpts,d_scanblockstartpts);
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " Kernel BinsStartPts_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
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
	cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

	timer.restart();
	checkCudaErrors(cudaMemcpy(&totalnupts,d_binstartpts+numbins[0]*numbins[1],sizeof(int),
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMalloc(&d_kxsorted,totalnupts*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_kysorted,totalnupts*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_csorted,totalnupts*sizeof(gpuComplex)));
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " Allocating GPU memory (need info of totolnupts) " << timer.elapsedsec() <<" s"<<endl;
#endif

	timer.restart();
	PtsRearrage_2d<<<(M+1024-1)/1024,1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0],
			numbins[1], d_binstartpts, d_sortidx, d_kx, d_kxsorted,
			d_ky, d_kysorted, d_c, d_csorted);
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " Kernel PtsRearrange_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
	FLT *h_kxsorted, *h_kysorted;
	CPX *h_csorted;
	h_kxsorted = (FLT*)malloc(totalnupts*sizeof(FLT));
	h_kysorted = (FLT*)malloc(totalnupts*sizeof(FLT));
	h_csorted  = (CPX*)malloc(totalnupts*sizeof(CPX));
	checkCudaErrors(cudaMemcpy(h_kxsorted,d_kxsorted,totalnupts*sizeof(FLT),
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_kysorted,d_kysorted,totalnupts*sizeof(FLT),
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_csorted,d_csorted,totalnupts*sizeof(CPX),
				cudaMemcpyDeviceToHost));
	for (int i=0; i<totalnupts; i++){
		//printf("[debug ] (x,y)=(%f, %f), bin#=%d\n", h_kxsorted[i], h_kysorted[i],
		//                                             (floor(h_kxsorted[i]/bin_size_x)+1)+numbins[0]*(floor(h_kysorted[i]/bin_size_y)+1));
		cout <<"[debug ] (x,y) = ("<<setw(10)<<h_kxsorted[i]<<","
			<<setw(10)<<h_kysorted[i]<<"), bin# =  "
			<<(floor(h_kxsorted[i]/bin_size_x)+1)+numbins[0]*(floor(h_kysorted[i]/bin_size_y)+1)<<endl;
	}
	free(h_kysorted);
	free(h_kxsorted);
	free(h_csorted);
#endif

	timer.restart();
	threadsPerBlock.x = 8;
	threadsPerBlock.y = 8;
	blocks.x = (nf1 + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = (nf2 + threadsPerBlock.y - 1)/threadsPerBlock.y;
	nbin_block_x = threadsPerBlock.x/bin_size_x<(numbins[0]-2) ? threadsPerBlock.x/bin_size_x : (numbins[0]-2);
	nbin_block_y = threadsPerBlock.y/bin_size_y<(numbins[1]-2) ? threadsPerBlock.y/bin_size_y : (numbins[1]-2);
#ifdef INFO
	cout<<"[info  ]"<<" ["<<nf1<<"x"<<nf2<<"] "<<"output elements is divided into ["
		<<blocks.x<<","<<blocks.y<<"] block"<<", each block has ["<<nbin_block_x<<"x"<<nbin_block_y<<"] bins, "
		<<"["<<threadsPerBlock.x<<"x"<<threadsPerBlock.y<<"] threads"<<endl;
#endif
	// blockSize must be a multiple of bin_size_x
	Spread_2d_Odriven<<<blocks, threadsPerBlock>>>(nbin_block_x, nbin_block_y, numbins[0], numbins[1],
			d_binstartpts, d_kxsorted, d_kysorted, d_csorted,
			d_fw, ns, nf1, nf2, es_c, es_beta, fw_width);
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " Kernel Spread_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
	timer.restart();
	//checkCudaErrors(cudaMemcpy(h_fw,d_fw,nf1*nf2*sizeof(gpuComplex),
	//                           cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(h_fw,nf1*sizeof(gpuComplex),d_fw,pitch,nf1*sizeof(gpuComplex),nf2,
				cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaDeviceSynchronize();
	cout<<"[time  ]"<< " Copying memory from device to host " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef TIME
	cout<<"[time  ]"<< " TOTAL SPREAD KERNEL TIME (exclude memalloc, memcpy): " << k_spread_time <<" s"<<endl;
#endif
	// Free memory
	cudaFree(d_kx);
	cudaFree(d_ky);
	cudaFree(d_c);
	cudaFree(d_fw);
	cudaFree(d_binsize);
	cudaFree(d_binstartpts);
	cudaFree(d_sortidx);
	cudaFree(d_kxsorted);
	cudaFree(d_kysorted);
	cudaFree(d_csorted);
	free(h_binsize);
	free(h_binstartpts);
	free(h_sortidx);
	return 0;
}

int cnufftspread2d_gpu_idriven(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX *h_c, spread_opts opts)
{
	CNTime timer;
	FLT k_spread_time=0.0;
	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;

	FLT *d_kx,*d_ky;
	gpuComplex *d_c, *d_fw;

	timer.restart();
	checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(gpuComplex)));
	//checkCudaErrors(cudaMalloc(&d_fw,2*nf1*nf2*sizeof(FLT)));
	int fw_width;
	size_t pitch;
	checkCudaErrors(cudaMallocPitch((void**) &d_fw,&pitch,nf1*sizeof(gpuComplex),nf2));
	fw_width = pitch/sizeof(gpuComplex);
#ifdef TIME
	cout<<"[time  ]"<< " Allocating GPU memory " << timer.elapsedsec() <<" s"<<endl;
#endif

	timer.restart();
	checkCudaErrors(cudaMemcpy(d_kx,h_kx,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ky,h_ky,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c,h_c,M*sizeof(gpuComplex),cudaMemcpyHostToDevice));
#ifdef TIME
	cout<<"[time  ]"<< " Copying memory from host to device " << timer.elapsedsec() <<" s"<<endl;
#endif

	timer.restart();
	threadsPerBlock.x = 1024;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	Spread_2d_Idriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c, d_fw, M, ns,
			nf1, nf2, es_c, es_beta, fw_width);
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " Kernel Spread_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
	timer.restart();
	//checkCudaErrors(cudaMemcpy(h_fw,d_fw,2*nf1*nf2*sizeof(FLT),
	//                           cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(h_fw,nf1*sizeof(gpuComplex),d_fw,pitch,nf1*sizeof(gpuComplex),nf2,
				cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaDeviceSynchronize();
	cout<<"[time  ]"<< " Copying memory from device to host " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef TIME
	cout<<"[time  ]"<< " TOTAL SPREAD KERNEL TIME (exclude memalloc, memcpy): " << k_spread_time <<" s"<<endl;
#endif

	// Free memory
	cudaFree(d_kx);
	cudaFree(d_ky);
	cudaFree(d_c);
	cudaFree(d_fw);
	return 0;
}

int cnufftspread2d_gpu_idriven_sorted(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX *h_c, spread_opts opts)
{
	CNTime timer;
	FLT k_spread_time=0.0;
	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;

	FLT *d_kx,*d_ky;
	gpuComplex *d_c, *d_fw;
	FLT *d_kxsorted,*d_kysorted;
	gpuComplex *d_csorted;
	int *d_sortidx;

	timer.restart();
	checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(gpuComplex)));

	checkCudaErrors(cudaMalloc(&d_kxsorted,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_kysorted,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_csorted,M*sizeof(gpuComplex)));

	checkCudaErrors(cudaMalloc(&d_sortidx,M*sizeof(int)));
	int fw_width;
	size_t pitch;
	checkCudaErrors(cudaMallocPitch((void**) &d_fw,&pitch,nf1*sizeof(gpuComplex),nf2));
	fw_width = pitch/sizeof(gpuComplex);
#ifdef TIME
	cout<<"[time  ]"<< " Allocating GPU memory " << timer.elapsedsec() <<" s"<<endl;
#endif

	timer.restart();
	checkCudaErrors(cudaMemcpy(d_kx,h_kx,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ky,h_ky,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c,h_c,M*sizeof(gpuComplex),cudaMemcpyHostToDevice));
#ifdef TIME
	cout<<"[time  ]"<< " Copying memory from host to device " << timer.elapsedsec() <<" s"<<endl;
#endif

	timer.restart();
	threadsPerBlock.x = 1024;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	CreateSortIdx<<<blocks, threadsPerBlock>>>(M, nf1, nf2, d_kx, d_ky, d_sortidx);
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " CreateSortIdx " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
	int* h_sortidx = (int*) malloc(M*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		printf("sortidx = %d, (x,y) = (%.3g, %.3g), c=(%f, %f)\n", h_sortidx[i], h_kx[i], h_ky[i], h_c[i].real(), h_c[i].imag());
	}
	free(h_sortidx);
#endif 
	if(opts.use_thrust){
		timer.restart();
		thrust::counting_iterator<int> iter(0);
		thrust::device_vector<int> indices(M);
		thrust::copy(iter, iter + indices.size(), indices.begin());
		thrust::sort_by_key(thrust::device, d_sortidx, d_sortidx + M, indices.begin());
#ifdef TIME
		cudaDeviceSynchronize();
		k_spread_time+=timer.elapsedsec();
		cout<<"[time  ]"<< " thrust::sort_by_key " << timer.elapsedsec() <<" s"<<endl;
#endif
		timer.restart();
		thrust::gather(thrust::device, indices.begin(), indices.end(), d_kx, d_kxsorted);
		thrust::gather(thrust::device, indices.begin(), indices.end(), d_ky, d_kysorted);
		thrust::gather(thrust::device, indices.begin(), indices.end(), d_c, d_csorted);
#ifdef TIME
		cudaDeviceSynchronize();
		cout<<"[time  ]"<< " thrust::gather " << timer.elapsedsec() <<" s"<<endl;
#endif
	}else{
		timer.restart();
		size_t  temp_storage_bytes  = 0;
		void    *d_temp_storage     = NULL;

		int *d_sortedidx;
		checkCudaErrors(cudaMalloc(&d_sortedidx,M*sizeof(int)));
		int *d_index_out, *d_index_in;
		checkCudaErrors(cudaMalloc(&d_index_in,M*sizeof(int)));
		checkCudaErrors(cudaMalloc(&d_index_out,M*sizeof(int)));

		threadsPerBlock.x = 1024;
		threadsPerBlock.y = 1;
		blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
		blocks.y = 1;
		CreateIndex<<<blocks, threadsPerBlock>>>(d_index_in, M);
		//#ifdef TIME
		//cudaDeviceSynchronize();
		//cout<<"[time  ]"<< " CreateIndex " << timer.elapsedsec() <<" s"<<endl;
		//#endif

		timer.restart();
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortidx, d_sortedidx, d_index_in, d_index_out, M);
		checkCudaErrors(cudaMalloc(&d_temp_storage,temp_storage_bytes));
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sortidx, d_sortedidx, d_index_in, d_index_out, M);

#ifdef TIME
		cudaDeviceSynchronize();
		k_spread_time+=timer.elapsedsec();
		cout<<"[time  ]"<< " cub::SortPairs " << timer.elapsedsec() <<" s"<<endl;
#endif
		timer.restart();
		threadsPerBlock.x = 1024;
		threadsPerBlock.y = 1;
		blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
		blocks.y = 1;
		Gather<<<blocks, threadsPerBlock>>>(M, d_index_out, d_kx, d_ky, d_c, d_kxsorted, d_kysorted, d_csorted);
		//thrust::gather(thrust::device, d_index_out, d_index_out+M, d_kx, d_kxsorted);
		//thrust::gather(thrust::device, d_index_out, d_index_out+M, d_ky, d_kysorted);
		//thrust::gather(thrust::device, d_index_out, d_index_out+M, d_c, d_csorted);
#ifdef TIME
		cudaDeviceSynchronize();
		k_spread_time+=timer.elapsedsec();
		cout<<"[time  ]"<< " Gather kernel " << timer.elapsedsec() <<" s"<<endl;
#endif
	}
#ifdef DEBUG
	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_kx,d_kxsorted,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_ky,d_kysorted,M*sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_c,d_csorted,M*sizeof(gpuComplex),cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		printf("sortidx = %d, (x,y) = (%.3g, %.3g), c=(%f, %f)\n", h_sortidx[i], h_kx[i], h_ky[i], h_c[i].real(), h_c[i].imag());
	}
#endif 

	timer.restart();
	threadsPerBlock.x = 1024;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;
	Spread_2d_Idriven<<<blocks, threadsPerBlock>>>(d_kxsorted, d_kysorted, d_csorted, d_fw, M, ns,
			nf1, nf2, es_c, es_beta, fw_width);
#ifdef TIME
	cudaDeviceSynchronize();
	k_spread_time+=timer.elapsedsec();
	cout<<"[time  ]"<< " Kernel Spread_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
	timer.restart();
	//checkCudaErrors(cudaMemcpy(h_fw,d_fw,2*nf1*nf2*sizeof(FLT),
	//                           cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(h_fw,nf1*sizeof(gpuComplex),d_fw,pitch,nf1*sizeof(gpuComplex),nf2,
				cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaDeviceSynchronize();
	cout<<"[time  ]"<< " Copying memory from device to host " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef TIME
	cout<<"[time  ]"<< " TOTAL SPREAD KERNEL TIME (exclude memalloc, memcpy): " << k_spread_time <<" s"<<endl;
#endif
	// Free memory
	cudaFree(d_kx);
	cudaFree(d_ky);
	cudaFree(d_c);
	cudaFree(d_kxsorted);
	cudaFree(d_kysorted);
	cudaFree(d_csorted);
	cudaFree(d_fw);
	cudaFree(d_sortidx);
	return 0;
}

int cnufftspread2d_gpu_hybrid(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
		FLT *h_ky, CPX *h_c, spread_opts opts)
{
	CNTime timer;
	FLT k_spread_time=0.0;
	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=opts.nspread;   // psi's support in terms of number of cells
	FLT es_c=opts.ES_c;
	FLT es_beta=opts.ES_beta;
	int bin_size_x=opts.bin_size_x;
	int bin_size_y=opts.bin_size_y;

	FLT *d_kx,*d_ky;
	gpuComplex *d_c,*d_fw;
	// Parameter setting
	int numbins[2];

	int *d_binsize;
	int *d_binstartpts;
	int *d_sortidx;

	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
	// assume that bin_size_x > ns/2;
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<opts.bin_size_x<<"x"<<opts.bin_size_y<<"]"<<endl;
	cout<<"[info  ] numbins (including ghost bins) = ["
		<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif
	FLT *d_kxsorted,*d_kysorted;
	gpuComplex *d_csorted;
	int *h_binsize, *h_binstartpts, *h_sortidx; // For debug

	timer.restart();
	checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c ,M*sizeof(gpuComplex)));
	//checkCudaErrors(cudaMalloc(&d_fw,nf1*nf2*sizeof(gpuComplex)));
	int fw_width;
	size_t pitch;
	checkCudaErrors(cudaMallocPitch((void**) &d_fw, &pitch,nf1*sizeof(gpuComplex),nf2));
	fw_width = pitch/sizeof(gpuComplex);

	checkCudaErrors(cudaMalloc(&d_binsize,numbins[0]*numbins[1]*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_sortidx,M*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_binstartpts,(numbins[0]*numbins[1]+1)*sizeof(int)));
#ifdef TIME
	cout<<"[time  ]"<< " Allocating GPU memory " << timer.elapsedsec() <<" s"<<endl;
#endif

	timer.restart();
	checkCudaErrors(cudaMemcpy(d_kx,h_kx,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_ky,h_ky,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c,h_c,M*sizeof(gpuComplex),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(&d_kxsorted,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_kysorted,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_csorted,M*sizeof(gpuComplex)));
#ifdef TIME
	cout<<"[time  ]"<< " Copying memory from host to device " << timer.elapsedsec() <<" s"<<endl;
#endif

	h_binsize     = (int*)malloc(numbins[0]*numbins[1]*sizeof(int));
	h_sortidx     = (int*)malloc(M*sizeof(int));
	h_binstartpts = (int*)malloc((numbins[0]*numbins[1]+1)*sizeof(int));
	checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*sizeof(int)));
	timer.restart();
	CalcBinSize_noghost_2d<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,bin_size_x,bin_size_y,
			numbins[0],numbins[1],d_binsize,
			d_kx,d_ky,d_sortidx);
#ifdef TIME
	cudaDeviceSynchronize();
	cout<<"[time  ]"<< " Kernel CalcBinSize_noghost_2d (#blocks, #threads)=("<<(M+1024-1)/1024<<","<<1024<<") takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
	checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*sizeof(int),
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
				cudaMemcpyDeviceToHost));
	cout<<"[debug ] Before fill in the ghost bin size:"<<endl;
	for(int j=0; j<numbins[1]; j++){
		cout<<"[debug ] ";
		for(int i=0; i<numbins[0]; i++){
			if(i!=0) cout<<" ";
			cout <<" bin["<<setw(3)<<i<<","<<setw(3)<<j<<"]="<<h_binsize[i+j*numbins[0]];
		}
		cout<<endl;
	}
	cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

	timer.restart();
	int n=numbins[0]*numbins[1];
	int scanblocksize=1024;
	int numscanblocks=ceil((double)n/scanblocksize);
	int* d_scanblocksum, *d_scanblockstartpts;
#ifdef DEBUG
	printf("[debug ] n=%d, numscanblocks=%d\n",n,numscanblocks);
#endif 
	checkCudaErrors(cudaMalloc(&d_scanblocksum,numscanblocks*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_scanblockstartpts,(numscanblocks+1)*sizeof(int)));

	for(int i=0;i<numscanblocks;i++){
		int nelemtoscan=(n-scanblocksize*i)>scanblocksize ? scanblocksize : n-scanblocksize*i;
		prescan<<<1, scanblocksize/2>>>(nelemtoscan,d_binsize+i*scanblocksize,
				d_binstartpts+i*scanblocksize,d_scanblocksum+i);
	}
#ifdef DEBUG
	int* h_scanblocksum;
	h_scanblocksum     =(int*) malloc(numscanblocks*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_scanblocksum,d_scanblocksum,numscanblocks*sizeof(int),
				cudaMemcpyDeviceToHost));
	for(int i=0;i<numscanblocks;i++){
		cout<<"[debug ] scanblocksum["<<i<<"]="<<h_scanblocksum[i]<<endl;
	}
#endif
	int next = pow(2, ceil(log(numscanblocks+1)/log(2)));
	if(next > 2048){
		cout<<"error: number of elements to sort exceed the prescan capability"<<endl;
		return 1;
	}
	prescan<<<1, next/2>>>(numscanblocks,d_scanblocksum,d_scanblockstartpts,d_scanblockstartpts+numscanblocks);
#ifdef DEBUG
	int* h_scanblockstartpts = (int*) malloc((numscanblocks+1)*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_scanblockstartpts,d_scanblockstartpts,(numscanblocks+1)*sizeof(int),
				cudaMemcpyDeviceToHost));
	for(int i=0;i<numscanblocks+1;i++){
		cout<<"[debug ] scanblockstartpts["<<i<<"]="<<h_scanblockstartpts[i]<<endl;
	}
#endif
	uniformUpdate<<<numscanblocks,scanblocksize>>>(n,d_binstartpts,d_scanblockstartpts);
#ifdef TIME
	cudaDeviceSynchronize();
	cout<<"[time  ]"<< " Kernel BinsStartPts_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif

#ifdef DEBUG
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
	cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

	timer.restart();
	PtsRearrage_noghost_2d<<<(M+1024-1)/1024,1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0],
			numbins[1], d_binstartpts, d_sortidx, d_kx, d_kxsorted,
			d_ky, d_kysorted, d_c, d_csorted);
#ifdef TIME
	cudaDeviceSynchronize();
	printf("[time  ] #block=%d, #threads=%d\n", (M+1024-1)/1024,1024);
	cout<<"[time  ]"<< " Kernel PtsRearrange_noghost_2d takes " << timer.elapsedsec() <<" s"<<endl;
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
	for (int i=0; i<M; i++){
		//printf("[debug ] (x,y)=(%f, %f), bin#=%d\n", h_kxsorted[i], h_kysorted[i],
		//                                             (floor(h_kxsorted[i]/bin_size_x)+1)+numbins[0]*(floor(h_kysorted[i]/bin_size_y)+1));
		cout <<"[debug ] (x,y) = ("<<setw(10)<<h_kxsorted[i]<<","
			<<setw(10)<<h_kysorted[i]<<"), bin# =  "
			<<(floor(h_kxsorted[i]/bin_size_x))+numbins[0]*(floor(h_kysorted[i]/bin_size_y))<<endl;
	}
	free(h_kysorted);
	free(h_kxsorted);
	free(h_csorted);
#endif

	timer.restart();
	threadsPerBlock.x = 32;
	threadsPerBlock.y = 32;
	blocks.x = numbins[0];
	blocks.y = numbins[1];
	size_t sharedmemorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*sizeof(gpuComplex);
	if(sharedmemorysize > 49152){
		cout<<"error: not enought shared memory"<<endl;
		return 1;
	}
	// blockSize must be a multiple of bin_size_x
	Spread_2d_Hybrid<<<blocks, threadsPerBlock, sharedmemorysize>>>(d_kxsorted, d_kysorted, d_csorted, 
			d_fw, M, ns, nf1, nf2, 
			es_c, es_beta, fw_width, 
			d_binstartpts, d_binsize, 
			bin_size_x, bin_size_y);
#ifdef TIME
	cudaDeviceSynchronize();
	cout<<"[time  ]"<< " Kernel Spread_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
	timer.restart();
	//checkCudaErrors(cudaMemcpy(h_fw,d_fw,nf1*nf2*sizeof(gpuComplex),
	//                           cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(h_fw,nf1*sizeof(gpuComplex),d_fw,pitch,nf1*sizeof(gpuComplex),nf2,
				cudaMemcpyDeviceToHost));
#ifdef TIME
	cudaDeviceSynchronize();
	cout<<"[time  ]"<< " Copying memory from device to host " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef TIME
	cout<<"[time  ]"<< " TOTAL SPREAD KERNEL TIME (exclude memalloc, memcpy): " << k_spread_time <<" s"<<endl;
#endif
	// Free memory
	cudaFree(d_kx);
	cudaFree(d_ky);
	cudaFree(d_c);
	cudaFree(d_fw);
	cudaFree(d_binsize);
	cudaFree(d_binstartpts);
	cudaFree(d_sortidx);
	cudaFree(d_kxsorted);
	cudaFree(d_kysorted);
	cudaFree(d_csorted);
	free(h_binsize);
	free(h_binstartpts);
	free(h_sortidx);
	return 0;
}

