#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <cuComplex.h>
#include "../scan/scan_common.h"
#include "spread.h"

using namespace std;

#define INFO
#define DEBUG
#define RESULT
#define TIME

int cnufftspread2d_gpu_odriven(int nf1, int nf2, CPX* h_fw, int M, FLT *h_kx,
                               FLT *h_ky, CPX *h_c, int bin_size_x, int bin_size_y)
{
  CNTime timer;
  dim3 threadsPerBlock;
  dim3 blocks;

  FLT tol=1e-6;
  int ns=std::ceil(-log10(tol/10.0));   // psi's support in terms of number of cells
  FLT es_c=4.0/(ns*ns);
  FLT es_beta = 2.30 * (FLT)ns;

  FLT *d_kx,*d_ky;
  gpuComplex *d_c,*d_fw;
  // Parameter setting
  int numbins[2];
  int totalnupts;
  int nbin_block_x, nbin_block_y;

  int *d_binsize;
  int *d_binstartpts;
  int *d_sortidx;

  numbins[0] = ceil(nf1/bin_size_x)+2;
  numbins[1] = ceil(nf2/bin_size_y)+2;
  // assume that bin_size_x > ns/2;
#ifdef INFO
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
  checkCudaErrors(cudaMalloc(&d_fw,nf1*nf2*sizeof(gpuComplex)));
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
  CalcBinSize_2d<<<(M+128-1)/128, 128>>>(M,nf1,nf2,bin_size_x,bin_size_y,
                                      numbins[0],numbins[1],d_binsize,
                                      d_kx,d_ky,d_sortidx);
#ifdef TIME
  cudaDeviceSynchronize();
  cout<<"[time  ]"<< " Kernel CalcBinSize_2d (#blocks, #threads)=("<<(M+128-1)/128<<","<<128<<") takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBIG
  checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*numbins[1]*sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
                             cudaMemcpyDeviceToHost));
  cout<<"[debug ] Before fill in the ghost bin size:"<<endl;
  for(int j=0; j<numbins[1]; j++){
    cout<<"[debug ] ";
    for(int i=0; i<numbins[0]; i++){
      if(i!=0) cout<<" ";
      cout <<"bin["<<i<<","<<j<<"] = "<<h_binsize[i+j*numbins[0]];
    }
    cout<<endl;
  }
  cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif
  timer.restart();
  threadsPerBlock.x = 32;
  threadsPerBlock.y = 32;// doesn't work for 64, doesn't know why
  if(threadsPerBlock.x*threadsPerBlock.y < 1024){
    cout<<"number of threads in a block exceeds max num 1024("
        <<threadsPerBlock.x*threadsPerBlock.y<<")"<<endl;
    return 1;
  }
  blocks.x = (numbins[0]+threadsPerBlock.x-1)/threadsPerBlock.x;
  blocks.y = (numbins[1]+threadsPerBlock.y-1)/threadsPerBlock.y;
  FillGhostBin_2d<<<blocks,threadsPerBlock>>>(numbins[0],numbins[1],d_binsize);
#ifdef TIME
  cudaDeviceSynchronize();
  printf("[time  ] block=(%d, %d), threads=(%d, %d)\n", blocks.x, blocks.y, threadsPerBlock.x, threadsPerBlock.y);
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
      cout <<"bin["<<i<<","<<j<<"] = "<<h_binsize[i+j*numbins[0]];
    }
    cout<<endl;
  }
  cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

  timer.restart();
  int n=numbins[0]*numbins[1];
  int scanblocksize=512;
  int numscanblocks=ceil((double)n/scanblocksize);
  int* d_scanblocksum, *d_scanblockstartpts;
  int* h_scanblocksum, *h_scanblockstartpts;
#ifdef TIME
  printf("[debug ] n=%d, numscanblocks=%d\n",n,numscanblocks);
#endif 
  checkCudaErrors(cudaMalloc(&d_scanblocksum,numscanblocks*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_scanblockstartpts,(numscanblocks+1)*sizeof(int)));
  h_scanblocksum     =(int*) malloc(numscanblocks*sizeof(int));
  h_scanblockstartpts=(int*) malloc((numscanblocks+1)*sizeof(int));
 
  for(int i=0;i<numscanblocks;i++){
    int nelemtoscan=(n-scanblocksize*i)>scanblocksize ? scanblocksize : n-scanblocksize*i;
    prescan<<<1, scanblocksize/2>>>(nelemtoscan,d_binsize+i*scanblocksize,
			            d_binstartpts+i*scanblocksize,d_scanblocksum+i);
  }
  checkCudaErrors(cudaMemcpy(h_scanblocksum,d_scanblocksum,numscanblocks*sizeof(int),
		             cudaMemcpyDeviceToHost));
  h_scanblockstartpts[0] = 0;
  for(int i=1;i<numscanblocks+1;i++){
    h_scanblockstartpts[i] = h_scanblockstartpts[i-1]+h_scanblocksum[i-1];
  }
#ifdef DEBUG
  for(int i=0;i<numscanblocks+1;i++){
    cout<<"[debug ] scanblocksum["<<i<<"]="<<h_scanblockstartpts[i]<<endl;
  }
#endif
  checkCudaErrors(cudaMemcpy(d_scanblockstartpts,h_scanblockstartpts,(numscanblocks+1)*sizeof(int),
		             cudaMemcpyHostToDevice));
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
      cout <<"bin["<<i<<","<<j<<"] = "<<setw(2)<<h_binstartpts[i+j*numbins[0]];
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
  cout<<"[time  ]"<< " Allocating GPU memory (need info of totolnupts) " << timer.elapsedsec() <<" s"<<endl;
#endif

  timer.restart();
  PtsRearrage_2d<<<(M+1024-1)/1024,1024>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0],
                                      numbins[1], d_binstartpts, d_sortidx, d_kx, d_kxsorted,
                                      d_ky, d_kysorted, d_c, d_csorted);
#ifdef TIME
  cudaDeviceSynchronize();
  printf("[time  ] #block=%d, #threads=%d\n", (M+1024-1)/1024,1024);
  cout<<"[time  ]"<< " Kernel PtsRearrange_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#if 1
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
                                                 d_fw, ns, nf1, nf2, es_c, es_beta);
#ifdef TIME
  cudaDeviceSynchronize();
  cout<<"[time  ]"<< " Kernel Spread_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
  timer.restart();
  checkCudaErrors(cudaMemcpy(h_fw,d_fw,nf1*nf2*sizeof(gpuComplex),
                             cudaMemcpyDeviceToHost));
#ifdef TIME
  cudaDeviceSynchronize();
  cout<<"[time  ]"<< " Copying memory from device to host " << timer.elapsedsec() <<" s"<<endl;
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

int cnufftspread2d_gpu_idriven(int nf1, int nf2, FLT* h_fw, int M, FLT *h_kx,
                               FLT *h_ky, FLT *h_c)
{
  CNTime timer;
  dim3 threadsPerBlock;
  dim3 blocks;

  FLT tol=1e-6;
  int ns=std::ceil(-log10(tol/10.0));   // psi's support in terms of number of cells
  FLT es_c=4.0/(ns*ns);
  FLT es_beta = 2.30 * (FLT)ns;

  FLT *d_c,*d_kx,*d_ky,*d_fw;

  timer.restart();
  checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(FLT)));
  checkCudaErrors(cudaMalloc(&d_ky,M*sizeof(FLT)));
  checkCudaErrors(cudaMalloc(&d_c,2*M*sizeof(FLT)));
  checkCudaErrors(cudaMalloc(&d_fw,2*nf1*nf2*sizeof(FLT)));
#ifdef TIME
  cout<<"[time  ]"<< " Allocating GPU memory " << timer.elapsedsec() <<" s"<<endl;
#endif

  timer.restart();
  checkCudaErrors(cudaMemcpy(d_kx,h_kx,M*sizeof(FLT),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ky,h_ky,M*sizeof(FLT),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_c,h_c,2*M*sizeof(FLT),cudaMemcpyHostToDevice));
#ifdef TIME
  cout<<"[time  ]"<< " Copying memory from host to device " << timer.elapsedsec() <<" s"<<endl;
#endif

  timer.restart();
  threadsPerBlock.x = 1024;
  threadsPerBlock.y = 1;
  blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
  blocks.y = 1;
  Spread_2d_Idriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, d_c, d_fw, M, ns,
                                                 nf1, nf2, es_c, es_beta);
#ifdef TIME
  cudaDeviceSynchronize();
  cout<<"[time  ]"<< " Kernel Spread_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
  timer.restart();
  checkCudaErrors(cudaMemcpy(h_fw,d_fw,2*nf1*nf2*sizeof(FLT),
                             cudaMemcpyDeviceToHost));
#ifdef TIME
  cudaDeviceSynchronize();
  cout<<"[time  ]"<< " Copying memory from device to host " << timer.elapsedsec() <<" s"<<endl;
#endif

// Free memory
  cudaFree(d_kx);
  cudaFree(d_ky);
  cudaFree(d_c);
  cudaFree(d_fw);
  return 0;
}
