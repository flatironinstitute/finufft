#include "spread.h"

int cnufftspread2d_gpu_odriven(int nf1, int nf2, FLT* h_fw, int M, FLT *h_kx,
                               FLT *h_ky, FLT *h_c, int bin_size_x, int bin_size_y)
{
  CNTime timer;
  dim3 threadsPerBlock;
  dim3 blocks;

  FLT tol=1e-6;
  int ns=std::ceil(-log10(tol/10.0));   // psi's support in terms of number of cells
  int es_c=4.0/(ns*ns);
  FLT es_beta = 2.30 * (FLT)ns;

  FLT *d_c,*d_kx,*d_ky,*d_fw;
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
  cout<<"[info  ] --> numbins (including ghost bins) = ["
      <<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif
  FLT *d_kxsorted,*d_kysorted,*d_csorted;
  int *h_binsize, *h_binstartpts, *h_sortidx; // For debug

  timer.restart();
  checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(FLT)));
  checkCudaErrors(cudaMalloc(&d_ky,M*sizeof(FLT)));
  checkCudaErrors(cudaMalloc(&d_c,2*M*sizeof(FLT)));
  checkCudaErrors(cudaMalloc(&d_fw,2*nf1*nf2*sizeof(FLT)));
  checkCudaErrors(cudaMalloc(&d_binsize,numbins[0]*numbins[1]*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_sortidx,M*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_binstartpts,(numbins[0]*numbins[1]+1)*sizeof(int)));
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

  h_binsize     = (int*)malloc(numbins[0]*numbins[1]*sizeof(int));
  h_sortidx     = (int*)malloc(M*sizeof(int));
  h_binstartpts = (int*)malloc((numbins[0]*numbins[1]+1)*sizeof(int));
  checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*sizeof(int)));
  timer.restart();
  CalcBinSize_2d<<<64, (M+64-1)/64>>>(M,nf1,nf2,bin_size_x,bin_size_y,
                                      numbins[0],numbins[1],d_binsize,
                                      d_kx,d_ky,d_sortidx);
#ifdef TIME
  cudaDeviceSynchronize();
  cout<<"[time  ]"<< " Kernel CalcBinSize_2d  takes " << timer.elapsedsec() <<" s"<<endl;
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
      cout <<"bin["<<i<<","<<j<<"] = "<<h_binsize[i+j*numbins[0]];
    }
    cout<<endl;
  }
  cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif
  timer.restart();
  threadsPerBlock.x = 16;
  threadsPerBlock.y = 16;
  blocks.x = (numbins[0]+threadsPerBlock.x-1)/threadsPerBlock.x;
  blocks.y = (numbins[1]+threadsPerBlock.y-1)/threadsPerBlock.y;
  FillGhostBin_2d<<<blocks, threadsPerBlock>>>(bin_size_x, bin_size_y, numbins[0],
                                               numbins[1], d_binsize);
#ifdef TIME
  cudaDeviceSynchronize();
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
  int THREADBLOCK_SIZE=256;
  int MIN_LARGE_ARRAY_SIZE=THREADBLOCK_SIZE*8;
  int arrayLength=THREADBLOCK_SIZE*8;
  if(numbins[0]*numbins[1] < MIN_LARGE_ARRAY_SIZE){ // 1024 is the maximum #threads per block
    int arrayLength=THREADBLOCK_SIZE*4;
    int szWorkgroup = scanExclusiveShort(d_binstartpts, d_binsize, numbins[0]*numbins[1]/ arrayLength, arrayLength);
    //BinsStartPts_2d<<<1, numbins[0]*numbins[1]>>>(M,numbins[0]*numbins[1],
    //                                              d_binsize,d_binstartpts);
    //prescan<<<1, numbins[0]*numbins[1]/2, numbins[0]*numbins[1]>>>(numbins[0]*numbins[1],
    //                                                               d_binsize,
    //                                                               d_binstartpts);
  }else{
    int szWorkgroup = scanExclusiveLarge(d_binstartpts, d_binsize, numbins[0]*numbins[1]/ arrayLength, arrayLength);
    cout<<"number of bins can't fit in one block"<<endl;
    return 1;
  }
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
  checkCudaErrors(cudaMalloc(&d_csorted, 2*totalnupts*sizeof(FLT)));
#ifdef TIME
  cudaDeviceSynchronize();
  cout<<"[time  ]"<< " Allocating GPU memory (need info of totolnupts) " << timer.elapsedsec() <<" s"<<endl;
#endif

  timer.restart();
  PtsRearrage_2d<<<64, (M+64-1)/64>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0],
                                      numbins[1], d_binstartpts, d_sortidx, d_kx, d_kxsorted,
                                      d_ky, d_kysorted, d_c, d_csorted);
#ifdef TIME
  cudaDeviceSynchronize();
  cout<<"[time  ]"<< " Kernel PtsRearrange_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
  FLT *h_kxsorted, *h_kysorted, *h_csorted;
  h_kxsorted = (FLT*)malloc(totalnupts*sizeof(FLT));
  h_kysorted = (FLT*)malloc(totalnupts*sizeof(FLT));
  h_csorted  = (FLT*)malloc(2*totalnupts*sizeof(FLT));
  checkCudaErrors(cudaMemcpy(h_kxsorted,d_kxsorted,totalnupts*sizeof(FLT),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_kysorted,d_kysorted,totalnupts*sizeof(FLT),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_csorted,d_csorted,2*totalnupts*sizeof(FLT),
                             cudaMemcpyDeviceToHost));
  for (int i=0; i<totalnupts; i++){
    cout <<"[debug ] (x,y) = ("<<setw(10)<<h_kxsorted[i]<<","
         <<setw(10)<<h_kysorted[i]<<"), bin# =  "
         <<(floor(h_kxsorted[i]/bin_size_x)+1)+numbins[0]*(floor(h_kysorted[i]/bin_size_y)+1)<<endl;
  }
  free(h_kysorted);
  free(h_kxsorted);
  free(h_csorted);
#endif

  timer.restart();
  threadsPerBlock.x = 32;
  threadsPerBlock.y = 32;
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
  int es_c=4.0/(ns*ns);
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
  threadsPerBlock.x = 64;
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
