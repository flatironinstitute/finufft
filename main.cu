#include <iostream>
#include <math.h>
#include <cuComplex.h>
#include <helper_cuda.h>
#include "spread1d.h"

using namespace std;

#define rand01() ((double)rand()/RAND_MAX)
// unif[-1,1]:
#define randm11() (2*rand01() - (double)1.0)
#define M_1_2PI 0.159154943091895336
#define max_shared_mem 6000

int main(void)
{
  // Parameter setting
  int N1 = 16;
  int nf1;
  int M = 1;
  int numbins;
  int bin_size_x = 32;
  int firstbinsize, lastbinsize;
  int blockSize, numblocks;
  int numbinperblock;

  unsigned int *d_binsize;
  unsigned int *d_binstartpts;
  unsigned int *d_sortidx;


  double sigma = 2;
  double *x, *xsorted;
  double tol=1e-6;

  int ns = std::ceil(-log10(tol/10.0));   // psi's support in terms of number of cells
  int es_c = 4.0/(ns*ns);  
  double es_beta = 2.30 * (double)ns;

  nf1 = (int) sigma*N1;
  numbins = ceil(nf1/bin_size_x); // assume that bin_size_x > ns/2;

  cuDoubleComplex *c, *csorted, *fw;
  unsigned int *h_binsize, *h_binstartpts, *h_sortidx; // For debug
  
  // Allocate Unified Memory – accessible from CPU or GPU
  checkCudaErrors(cudaMallocManaged(&x, M*sizeof(double)));
  checkCudaErrors(cudaMallocManaged(&c, M*sizeof(cuDoubleComplex)));
  checkCudaErrors(cudaMallocManaged(&fw, nf1*sizeof(cuDoubleComplex)));
  
  // Allocate GPU arrays
  h_binsize     = (unsigned int*)malloc(numbins*sizeof(unsigned int));
  h_binstartpts = (unsigned int*)malloc((numbins+3)*sizeof(unsigned int));
  h_sortidx     = (unsigned int*)malloc(M*sizeof(unsigned int));
  
  checkCudaErrors(cudaMalloc(&d_binsize    , numbins*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_binstartpts, (numbins+3)*sizeof(unsigned int)));//include ghost bins
  checkCudaErrors(cudaMalloc(&d_sortidx    , M*sizeof(int)));
  
  // initialize binsize array
  checkCudaErrors(cudaMemset(d_binsize, 0, numbins*sizeof(unsigned int)));
  // initialize x and y arrays on the host
  for (int i = 0; i < M; i++) {
    x[i] = M_PI*randm11();// x in [-pi,pi)
    c[i] = make_cuDoubleComplex(randm11(),randm11());
  }

  blockSize = 64;
  numblocks = (M + blockSize - 1)/blockSize;

  CalcBinSize<<<numblocks, blockSize>>>(M, nf1, bin_size_x, d_binsize, x, d_sortidx);
  BinsStartPts<<<1, numbins, numbins>>>(M, numbins, d_binsize, d_binstartpts);

  checkCudaErrors(cudaMemcpy(h_binsize,     d_binsize,     numbins*sizeof(unsigned int)    , cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_binstartpts, d_binstartpts, (numbins+3)*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_sortidx,     d_sortidx,     M*sizeof(unsigned int)          , cudaMemcpyDeviceToHost));
  
  firstbinsize = h_binsize[0];
  lastbinsize  = h_binsize[numbins-1];

  checkCudaErrors(cudaMallocManaged(&xsorted, (M+firstbinsize+lastbinsize)*sizeof(double)));
  checkCudaErrors(cudaMallocManaged(&csorted, (M+firstbinsize+lastbinsize)*sizeof(double)));
  PtsRearrage<<<numblocks, blockSize>>>(M, nf1, bin_size_x, numbins, d_binstartpts, d_sortidx, x, xsorted, c, csorted);
  
  blockSize = 64;
  numblocks = (nf1 + blockSize - 1)/blockSize;
  numbinperblock = blockSize/bin_size_x < numbins ? blockSize/bin_size_x : numbins; // blockSize must be a multiple of bin_size_x 
  Spread<<<numblocks, blockSize>>>(numbinperblock, d_binstartpts, xsorted,
                                   csorted, fw, ns, nf1, es_c, es_beta);
   
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
    cout <<"xsorted = "<<xsorted[i] <<endl;
  }

  for (int i=0; i<nf1; i++){
    if( i % bin_size_x == 0)
      cout<< "---------------------------------------" <<endl;
    cout <<"fw[" <<i <<"]="<<cuCreal(fw[i])<<endl;
  } 
#endif


  // Free memory
  cudaFree(d_binsize);
  cudaFree(d_binstartpts);
  cudaFree(d_sortidx);
  cudaFree(x);
  cudaFree(xsorted);
  cudaFree(c);
  cudaFree(csorted);
  cudaFree(fw);
  free(h_binsize); 
  free(h_binstartpts);
  free(h_sortidx);
  return 0;
}
