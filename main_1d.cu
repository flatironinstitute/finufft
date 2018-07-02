#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "spread.h"
#include "utils.h"

using namespace std;

//#define INFO
#define DEBUG
#define RESULT
#define TIME

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

int cnufftspread1d_gpu(int nf1, double* h_fw, int M, double *h_kx, double *h_c, int bin_size_x)
{
  CNTime timer;
  // Parameter setting
  int numbins[1];
  int totalnupts;
  int nbin_block_x;

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
#ifdef INFO
  cout<<"[info  ] --> numbins (including ghost bins) = ["
      <<numbins[0]<<"]"<<endl;
#endif
  // assume that bin_size_x > ns/2;

  double *d_c, *d_csorted, *d_fw;
  double *d_kx,*d_kxsorted;
  int *h_binsize, *h_binstartpts, *h_sortidx; // For debug

  timer.restart();
  checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_c,2*M*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_fw,2*nf1*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_binsize,numbins[0]*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_sortidx,M*sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_binstartpts,(numbins[0]+1)*sizeof(int)));
#ifdef TIME
  cout<<"[time  ]"<< " --- Allocating the GPU memory " << timer.elapsedsec() <<" s"<<endl;
#endif
  
  timer.restart();  
  checkCudaErrors(cudaMemcpy(d_kx,h_kx,M*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_c,h_c,2*M*sizeof(double),cudaMemcpyHostToDevice));
#ifdef TIME
  cout<<"[time  ]"<< " --- Copying memory from host to device " << timer.elapsedsec() <<" s"<<endl;
#endif
  
  h_binsize     = (int*)malloc(numbins[0]*sizeof(int));
  h_sortidx     = (int*)malloc(M*sizeof(int));
  h_binstartpts = (int*)malloc((numbins[0]+1)*sizeof(int));
  checkCudaErrors(cudaMemset(d_binsize,0,numbins[0]*sizeof(int)));
  timer.restart();
  CalcBinSize_1d<<<64, (M+64-1)/64>>>(M,nf1,bin_size_x,numbins[0],d_binsize,d_kx,d_sortidx);
#ifdef TIME
  cout<<"[time  ]"<< " Kernel CalcBinSize_1d  takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
  checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*sizeof(int), 
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_sortidx,d_sortidx,M*sizeof(int),
                             cudaMemcpyDeviceToHost));
  cout<<"[debug ] Before fill in the ghost bin size:"<<endl;
  for(int i=0; i<numbins[0]; i++){
    cout <<"[debug ] bin["<<i<<"] = "<<h_binsize[i];
    cout<<endl;
  }
  cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif
  timer.restart();
  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x = (numbins[0]+threadsPerBlock.x-1)/threadsPerBlock.x;
  blocks.y = 1;  
  FillGhostBin_1d<<<blocks, threadsPerBlock>>>(bin_size_x, numbins[0], d_binsize);
#ifdef TIME
  cout<<"[time  ]"<< " Kernel FillGhostBin_1d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
  checkCudaErrors(cudaMemcpy(h_binsize,d_binsize,numbins[0]*sizeof(int), 
                             cudaMemcpyDeviceToHost));
  cout<<"[debug ] After fill in the ghost bin size:"<<endl;
  for(int i=0; i<numbins[0]; i++){
    cout<<"[debug ] bin["<<i<<"] = "<<h_binsize[i];
    cout<<endl;
  }
  cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

  timer.restart();
  if(numbins[0] < 1024){ // 1024 is the maximum #threads per block 
    BinsStartPts_1d<<<1, numbins[0]>>>(M,numbins[0],d_binsize,d_binstartpts);
  }else{
    cout<<"number of bins can't fit in one block"<<endl;
    return 1;
  }
#ifdef TIME
  cout<<"[time  ]"<< " Kernel BinsStartPts_1d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG
  checkCudaErrors(cudaMemcpy(h_binstartpts,d_binstartpts,(numbins[0]+1)*sizeof(int), 
                             cudaMemcpyDeviceToHost));
  cout<<"[debug ] Result of scan bin_size array:"<<endl;
  for(int i=0; i<numbins[0]; i++){
    cout<<"[debug ] bin["<<i<<"] = "<<setw(2)<<h_binstartpts[i]<<endl;
  }
  cout<<"[debug ] Total number of nonuniform pts (include those in ghost bins) = "
      << setw(4)<<h_binstartpts[numbins[0]]<<endl;
  cout<<"[debug ] --------------------------------------------------------------"<<endl;
#endif

  timer.restart();
  checkCudaErrors(cudaMemcpy(&totalnupts,d_binstartpts+numbins[0],sizeof(int), 
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMalloc(&d_kxsorted,totalnupts*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_csorted, 2*totalnupts*sizeof(double)));
#ifdef TIME
  cout<<"[time  ]"<< " --- Allocating the GPU memory (need info of totolnupts) " << timer.elapsedsec() <<" s"<<endl;
#endif
  
  timer.restart();
  PtsRearrage_1d<<<64, (M+64-1)/64>>>(M, nf1, bin_size_x, numbins[0], 
                                      d_binstartpts, d_sortidx, d_kx, d_kxsorted, 
                                      d_c, d_csorted);
#ifdef TIME
  cout<<"[time  ]"<< " Kernel PtsRearrange_2d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
#ifdef DEBUG 
  double *h_kxsorted, *h_csorted;
  h_kxsorted = (double*)malloc(totalnupts*sizeof(double));
  h_csorted  = (double*)malloc(2*totalnupts*sizeof(double));
  checkCudaErrors(cudaMemcpy(h_kxsorted,d_kxsorted,totalnupts*sizeof(double),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_csorted,d_csorted,2*totalnupts*sizeof(double),
                             cudaMemcpyDeviceToHost));
  for (int i=0; i<totalnupts; i++){
    cout <<"[debug ] (x,y) = ("<<setw(10)<<h_kxsorted[i]<<")"<<", bin# =  "
         <<(floor(h_kxsorted[i]/bin_size_x)+1)<<endl;
  }
  free(h_kxsorted);
  free(h_csorted);
#endif
  
  timer.restart();
  threadsPerBlock.x = 32;
  blocks.x = (nf1 + threadsPerBlock.x - 1)/threadsPerBlock.x;
  nbin_block_x = threadsPerBlock.x/bin_size_x<(numbins[0]-2) ? threadsPerBlock.x/bin_size_x : (numbins[0]-2); 
#ifdef INFO
  cout<<"[info  ]"<<" ["<<nf1<<"] "<<"output elements is divided into ["
      <<blocks.x<<"] block"<<", each block has ["<<nbin_block_x<<"] bins, "
      <<"["<<threadsPerBlock.x<<"] threads"<<endl;
#endif
  // blockSize must be a multiple of bin_size_x 
  Spread_1d<<<blocks, threadsPerBlock>>>(nbin_block_x, numbins[0], d_binstartpts, d_kxsorted, 
                                         d_csorted, d_fw, ns, nf1, es_c, es_beta);
#ifdef TIME
  cout<<"[time  ]"<< " Kernel Spread_1d takes " << timer.elapsedsec() <<" s"<<endl;
#endif
  timer.restart();
  checkCudaErrors(cudaMemcpy(h_fw,d_fw,2*nf1*sizeof(double),
                             cudaMemcpyDeviceToHost));
#ifdef TIME
  cout<<"[time  ]"<< " --- Copying memory from device to host " << timer.elapsedsec() <<" s"<<endl;
#endif
  
// Free memory
  cudaFree(d_binsize);
  cudaFree(d_binstartpts);
  cudaFree(d_sortidx);
  cudaFree(d_kx);
  cudaFree(d_kxsorted);
  cudaFree(d_c);
  cudaFree(d_csorted);
  cudaFree(d_fw);
  free(h_binsize); 
  free(h_binstartpts);
  free(h_sortidx);
  return 0;
}

int main(int argc, char* argv[])
{
  cout<<setprecision(3)<<endl;
  int N1 = 16;
  int M = 1;
  double sigma = 2.0;
  int bin_size_x = 4;
  int nf1 = (int) sigma*N1;
  
  double *x;
  complex<double> *c, *fw;
  x  = (double*) malloc(M*sizeof(double));
  c  = (complex<double>*) malloc(M*sizeof(complex<double>));
  fw = (complex<double>*) malloc(nf1*sizeof(complex<double>));

  for (int i = 0; i < M; i++) {
    x[i] = M_PI*randm11();// x in [-pi,pi)
    c[i] = crandm11();
  }
#ifdef INFO
  cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"] uniform grids"<<endl;
  cout<<"[info  ] Dividing the uniform grids to bin size["<<bin_size_x<<"]"<<endl;
#endif
  CNTime timer; timer.start();
  int ier = cnufftspread1d_gpu(nf1, (double*) fw, M, x, (double*) c, bin_size_x);
  double ti=timer.elapsedsec();
#ifdef TIME
  printf("[info  ] %ld NU pts to (%ld) modes in %.3g s \t%.3g NU pts/s\n",M,N1,ti,M/ti);
#endif
#ifdef RESULT
  cout<<"[result]"<<endl;
    for (int i=0; i<nf1; i++){
      if( i % bin_size_x == 0 && i!=0)
        cout<< "--------" <<endl;
      //cout<<"fw[" <<i <<","<<j<<"]="<<fw[i+j*nf1];
      cout<<setw(5)<<fw[i]<<endl;
    }
  cout<<endl;
#endif
  free(x);
  free(c);
  free(fw);
  return 0;
}
