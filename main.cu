#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "spread.h"

using namespace std;

#define rand01() ((double)rand()/RAND_MAX)
// unif[-1,1]:
#define IMA complex<double>(0.0,1.0)
#define randm11() (2*rand01() - (double)1.0)
#define M_1_2PI 0.159154943091895336
#define max_shared_mem 6000
#define crandm11() (randm11() + IMA*randm11())

int cnufftspread1d_gpu(int nf1, double* h_fw, int M, double *h_kx, double *h_c, int bin_size_x)
{
  // Parameter setting
  int numbins;
  int firstbinsize, lastbinsize;
  int blockSize, numblocks;
  int numbinperblock;

  unsigned int *d_binsize;
  unsigned int *d_binstartpts;
  unsigned int *d_sortidx;

  double tol=1e-6;
  int ns=std::ceil(-log10(tol/10.0));   // psi's support in terms of number of cells
  int es_c=4.0/(ns*ns);  
  double es_beta = 2.30 * (double)ns;

  numbins = ceil(nf1/bin_size_x); // assume that bin_size_x > ns/2;

  double *d_c, *d_csorted, *d_fw;
  double *d_kx, *d_kxsorted;
  unsigned int *h_binsize, *h_binstartpts, *h_sortidx; // For debug
  double *h_kxsorted;

  // Allocate Unified Memory – accessible from CPU or GPU
  checkCudaErrors(cudaMalloc(&d_kx,M*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_c,2*M*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_fw,2*nf1*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_binsize,numbins*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_binstartpts,(numbins+3)*sizeof(unsigned int)));//include ghost bins
  checkCudaErrors(cudaMalloc(&d_sortidx, M*sizeof(int)));
  
  checkCudaErrors(cudaMemcpy(d_kx, h_kx, M*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_c, h_c, 2*M*sizeof(double), cudaMemcpyHostToDevice));
  // Allocate GPU arrays
  h_binsize     = (unsigned int*)malloc(numbins*sizeof(unsigned int));
  h_binstartpts = (unsigned int*)malloc((numbins+3)*sizeof(unsigned int));
  h_sortidx     = (unsigned int*)malloc(M*sizeof(unsigned int));
  
  
  // initialize binsize array
  checkCudaErrors(cudaMemset(d_binsize, 0, numbins*sizeof(unsigned int)));
  // initialize x and y arrays on the host

  blockSize = 64;
  numblocks = (M + blockSize - 1)/blockSize;

  CalcBinSize<<<numblocks, blockSize>>>(M, nf1, bin_size_x, d_binsize, d_kx, d_sortidx);
  BinsStartPts<<<1, numbins, numbins>>>(M, numbins, d_binsize, d_binstartpts);

  checkCudaErrors(cudaMemcpy(h_binsize,     d_binsize,     numbins*sizeof(unsigned int)    , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_binstartpts, d_binstartpts, (numbins+3)*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_sortidx,     d_sortidx,     M*sizeof(unsigned int)          , cudaMemcpyDeviceToHost));
  
  firstbinsize = h_binsize[0];
  lastbinsize  = h_binsize[numbins-1];

  cout << firstbinsize << endl;
  cout << lastbinsize << endl;
  checkCudaErrors(cudaMalloc(&d_kxsorted,   (M+firstbinsize+lastbinsize)*sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_csorted, 2*(M+firstbinsize+lastbinsize)*sizeof(double)));
  h_kxsorted    = (double*)malloc((M+firstbinsize+lastbinsize)*sizeof(double));
  PtsRearrage<<<numblocks, blockSize>>>(M, nf1, bin_size_x, numbins, d_binstartpts, d_sortidx, d_kx, d_kxsorted, d_c, d_csorted);
  
  blockSize = 64;
  numblocks = (nf1 + blockSize - 1)/blockSize;
  numbinperblock = blockSize/bin_size_x < numbins ? blockSize/bin_size_x : numbins; // blockSize must be a multiple of bin_size_x 
  Spread<<<numblocks, blockSize>>>(numbinperblock, d_binstartpts, d_kxsorted,
                                   d_csorted, d_fw, ns, nf1, es_c, es_beta);
   
  checkCudaErrors(cudaMemcpy(h_kxsorted, d_kxsorted, (M+firstbinsize+lastbinsize)*sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_fw, d_fw, 2*nf1*sizeof(double), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

#if 1
  for (int i=0; i<numbins+3; i++){
    cout << "binstartpts = " << h_binstartpts[i];
    if( i > 0 && i <= numbins)
      cout <<", binsize = "<<h_binsize[i-1];
    cout << endl;
  }
  int bin = 0;
  for (int i=0; i<M+lastbinsize+firstbinsize; i++){
    if ( i == h_binstartpts[bin] ){
      cout<< "---------------------------------------" << endl;
      bin++;
    }
    if ( h_binstartpts[bin] == h_binstartpts[bin-1]) bin++;
    cout <<"xsorted["<<i<<"] = "<<h_kxsorted[i] <<endl;
  }
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
  free(h_kxsorted);
  return 0;
}

int main()
{
  int N1 = 16;
  int M = 1;
  double sigma = 2.0;
  int bin_size_x = 32;
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
  int ier = cnufftspread1d_gpu(nf1, (double*) fw, M, x, (double*) c, bin_size_x);
  for (int i=0; i<nf1; i++){
    if( i % bin_size_x == 0)
      cout<< "---------------------------------------" <<endl;
    cout <<"fw[" <<i <<"]="<<fw[i]<<endl;
  } 
  free(x);
  free(c);
  free(fw);
  return 0;
}
