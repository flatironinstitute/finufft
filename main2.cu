#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "spread.h"
#include "utils.h"

using namespace std;

#define INFO
//#define DEBUG
//#define RESULT

#define rand01() ((double)rand()/RAND_MAX)
// unif[-1,1]:
#define IMA complex<double>(0.0,1.0)
#define randm11() (2*rand01() - (double)1.0)
#define crandm11() (randm11() + IMA*randm11())
#define PI (double)M_PI
#define M_1_2PI 0.159154943091895336
#define RESCALE(x,N,p) (p ? \
             ((x*M_1_2PI + (x<-PI ? 1.5 : (x>PI ? -0.5 : 0.5)))*N) : \
             (x<0 ? x+N : (x>N ? x-N : x)))

int cnufftspread2d_gpu(int nf1, int nf2, double* h_fw, int M, double *h_kx, 
                       double *h_ky, double *h_c, int bin_size_x, int bin_size_y)
{
  // Parameter setting
  int numbins[2];
  int totalnupts;
  int nbin_block_x, nbin_block_y;

  int *d_binsize;
  int *d_binstartpts;
  int *d_sortidx;

  double tol=1e-6;
  int ns=std::ceil(-log10(tol/10.0));   // psi's support in terms of number of cells
  int es_c=4.0/(ns*ns);  
  double es_beta = 2.30 * (double)ns;

  dim3 threadsPerBlock;
  dim3 blocks;
  
  numbins[0] = ceil(nf1/bin_size_x)+2;
  numbins[1] = ceil(nf2/bin_size_y)+2; 
#ifdef INFO
  cout<<"[info  ] --> numbins (including ghost bins) = ["
      <<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif
  // assume that bin_size_x > ns/2;

  double *d_c, *d_csorted, *d_fw;
  double *d_kx,*d_ky,*d_kxsorted,*d_kysorted;
  int *h_binsize, *h_binstartpts, *h_sortidx; // For debug
  double *h_kxsorted, *h_kysorted;

  // Allocate Unified Memory – accessible from CPU or GPU
  checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_ky,M*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_c,2*M*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_fw,2*nf1*nf2*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_binsize,numbins[0]*numbins[1]*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_sortidx,M*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_binstartpts,(numbins[0]*numbins[1]+1)*sizeof(int)));
  
  checkCudaErrors(cudaMemcpy(d_kx,h_kx,M*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ky,h_ky,M*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_c,h_c,2*M*sizeof(double),cudaMemcpyHostToDevice));
  
  h_binsize     = (int*)malloc(numbins[0]*numbins[1]*sizeof(int));
  h_sortidx     = (int*)malloc(M*sizeof(int));
  h_binstartpts = (int*)malloc((numbins[0]*numbins[1]+1)*sizeof(int));
  checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*numbins[1]*sizeof(int)));

  CalcBinSize_2d<<<64, (M+64-1)/64>>>(M,nf1,nf2,bin_size_x,bin_size_y,
                                      numbins[0],numbins[1],d_binsize,
                                      d_kx,d_ky,d_sortidx);
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

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 16;
  blocks.x = (numbins[0]+threadsPerBlock.x-1)/threadsPerBlock.x;
  blocks.y = (numbins[1]+threadsPerBlock.y-1)/threadsPerBlock.y;  
  FillGhostBin_2d<<<blocks, threadsPerBlock>>>(bin_size_x, bin_size_y, numbins[0], 
                                               numbins[1], d_binsize);
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
#if 1 
  if(numbins[0]*numbins[1] < 1024){ // 1024 is the maximum #threads per block 
    BinsStartPts_2d<<<1, numbins[0]*numbins[1]>>>(M,numbins[0]*numbins[1],
                                                  d_binsize,d_binstartpts);
  }else{
    cout<<"number of bins can't fit in one block"<<endl;
    return 1;
  }
#endif
  checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]*numbins[1]+1)*sizeof(int), 
                             cudaMemcpyDeviceToHost));
#ifdef DEBUG
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

  totalnupts = h_binstartpts[numbins[0]*numbins[1]];
  checkCudaErrors(cudaMalloc(&d_kxsorted,totalnupts*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_kysorted,totalnupts*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_csorted, totalnupts*sizeof(double)));
  h_kxsorted = (double*)malloc(totalnupts*sizeof(double));
  h_kysorted = (double*)malloc(totalnupts*sizeof(double));
  
  PtsRearrage_2d<<<64, (M+64-1)/64>>>(M, nf1, nf2, bin_size_x, bin_size_y, numbins[0], 
                                      numbins[1], d_binstartpts, d_sortidx, d_kx, d_kxsorted, 
                                      d_ky, d_kysorted, d_c, d_csorted);
#ifdef DEBUG 
  checkCudaErrors(cudaMemcpy(h_kxsorted,d_kxsorted,totalnupts*sizeof(double),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_kysorted,d_kysorted,totalnupts*sizeof(double),
                             cudaMemcpyDeviceToHost));
  for (int i=0; i<totalnupts; i++){
    cout <<"[debug ] (x,y) = ("<<setw(10)<<h_kxsorted[i]<<","
         <<setw(10)<<h_kysorted[i]<<"), bin# = "
         <<(floor(h_kxsorted[i]/bin_size_x)+1)+numbins[0]*(floor(h_kysorted[i]/bin_size_y)+1)<<endl;
  }
#endif
  
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
  Spread_2d<<<blocks, threadsPerBlock>>>(nbin_block_x, nbin_block_y, numbins[0], numbins[1], 
                                         d_binstartpts, d_kxsorted, d_kysorted, d_csorted, 
                                         d_fw, ns, nf1, nf2, es_c, es_beta);

  checkCudaErrors(cudaMemcpy(h_fw,d_fw,2*nf1*nf2*sizeof(double),
                             cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  
// Free memory
  cudaFree(d_binsize);
  cudaFree(d_binstartpts);
  cudaFree(d_sortidx);
  cudaFree(d_kx);
  cudaFree(d_ky);
  cudaFree(d_kxsorted);
  cudaFree(d_kysorted);
  cudaFree(d_c);
  cudaFree(d_csorted);
  cudaFree(d_fw);
  free(h_binsize); 
  free(h_binstartpts);
  free(h_sortidx);
  free(h_kxsorted);
  return 0;
}

int main(int argc, char* argv[])
{
  cout<<setprecision(3)<<endl;
  int N1 = 256, N2 = 256;
  int M = N1*N2;
  double sigma = 2.0;
  int bin_size_x = 32;
  int bin_size_y = 32;
  int nf1 = (int) sigma*N1;
  int nf2 = (int) sigma*N2;
  
  double *x, *y;
  complex<double> *c, *fw;
  x  = (double*) malloc(M*sizeof(double));
  y  = (double*) malloc(M*sizeof(double));
  c  = (complex<double>*) malloc(M*sizeof(complex<double>));
  fw = (complex<double>*) malloc(nf1*nf2*sizeof(complex<double>));

  for (int i = 0; i < M; i++) {
    x[i] = M_PI*randm11();// x in [-pi,pi)
    y[i] = M_PI*randm11();
    c[i] = crandm11();
  }
#ifdef INFO
  cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
  cout<<"[info  ] Dividing the uniform grids to bin size["<<bin_size_x<<"x"<<bin_size_y<<"]"<<endl;
#endif
  CNTime timer; timer.start();
  int ier = cnufftspread2d_gpu(nf1, nf2, (double*) fw, M, x, y,
                               (double*) c, bin_size_x, bin_size_y);
  double ti=timer.elapsedsec();
  printf("[info  ] %ld NU pts to (%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",M,N1,N2,ti,M/ti);
#ifdef RESULT
  cout<<"[result]"<<endl;
  for(int j=0; j<nf2; j++){
    if( j % bin_size_y == 0)
        cout<<endl;
    for (int i=0; i<nf1; i++){
      if( i % bin_size_x == 0 && i!=0)
        cout<< " |";
      //cout<<"fw[" <<i <<","<<j<<"]="<<fw[i+j*nf1];
      cout<<" "<<setw(8)<<fw[i+j*nf1];
    }
    cout<<endl;
  }
  cout<<endl;
#endif
  free(x);
  free(c);
  free(fw);
  return 0;
}
