#include <iostream>
#include <math.h>
#include <helper_cuda.h>

using namespace std;

#define rand01() ((double)rand()/RAND_MAX)
// unif[-1,1]:
#define randm11() (2*rand01() - (double)1.0)
#define M_1_2PI 0.159154943091895336
#define max_shared_mem 6000

__device__
double evaluate_kernel(double x, double es_c, double es_beta)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{
    return exp(es_beta * (sqrt(1.0 - es_c*x*x) - 1));
}

__global__
void CalcBinSize(int M, int nf1, int  bin_size_x, unsigned int* bin_size, double *x, unsigned int* sortidx)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  unsigned int binidx;
  int oldidx;
  double x_rescaled;
  if (i < M){
    x_rescaled = (x[i]*M_1_2PI + (x[i]<-M_PI ? 1.5 : (x[i]>M_PI ? -0.5 : 0.5)))*nf1;
    //x_rescaled = (x[i] > 0) ? x[i]*M_1_2PI*nf1 : (x[i]*M_1_2PI+1)*nf1; 
    binidx = floor(x_rescaled/bin_size_x);
    oldidx = atomicAdd(bin_size+binidx, 1);
    sortidx[i] = oldidx;
  }
}

// An exclusive scan of bin_size, only works for 1 block (!) 
__global__
void BinsStartPts(int M, int numbins, unsigned int* bin_size, unsigned int* bin_startpts)
{
  extern __shared__ unsigned int temp[];
  int lastbinsize = bin_size[numbins-1];
  int firstbinsize = bin_size[0];
  int i = threadIdx.x;
  temp[i] = (i > 0) ? bin_size[i-1] : 0;
  __syncthreads();
  for(int offset = 1; offset < numbins; offset*=2){
    if( i >= offset)
      temp[i] += temp[i - offset];
    else
      temp[i] = temp[i];
    __syncthreads();
  }
  bin_startpts[i+1] = temp[i] + lastbinsize;
  bin_startpts[0] = 0;
  bin_startpts[numbins+1] = M + lastbinsize;
  bin_startpts[numbins+2] = M + lastbinsize + firstbinsize;
}

__global__
void PtsRearrage(int M, int nf1, int bin_size_x, int numbins, unsigned int* bin_startpts, unsigned int* sortidx, 
                 double* x, double* x_sorted, 
                 double* c, double* c_sorted)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  unsigned int binidx;
  double x_rescaled;
  if( i < M){
    x_rescaled = (x[i]*M_1_2PI + (x[i]<-M_PI ? 1.5 : (x[i]>M_PI ? -0.5 : 0.5)))*nf1;
    binidx = floor(x_rescaled/bin_size_x);
    x_sorted[ bin_startpts[binidx+1]+sortidx[i]] = x_rescaled;
    c_sorted[ bin_startpts[binidx+1]+sortidx[i]] = c[i];
    if( binidx == 0 ){
      x_sorted[ bin_startpts[numbins+1]+sortidx[i] ] = x_rescaled + nf1;
      c_sorted[ bin_startpts[numbins+1]+sortidx[i] ] = c[i];
    }
    if( binidx == numbins-1){
      x_sorted[ sortidx[i] ] = x_rescaled - nf1;
      c_sorted[ sortidx[i] ] = c[i]; 
    }
  }
}

__global__
void Spread(unsigned int numbinperblock, unsigned int* bin_startpts, double* x_sorted, 
            double* c_sorted, double* fw, int ns, int nf1, double es_c,
            double es_beta)
{
  __shared__ double xshared[max_shared_mem/4];
  __shared__ double cshared[2*max_shared_mem/4];

  int i = blockDim.x*blockIdx.x + threadIdx.x;// output index, coord of the index
  int binxLo = blockIdx.x*numbinperblock;
  int binxHi = binxLo+numbinperblock+1;
  int start, end, j, bin;
  // run through all bins
  if( i < nf1 ){
    for(bin=binxLo; bin <= binxHi; bin++){
      start = bin_startpts[bin];
      end   = bin_startpts[bin+1];
      if( threadIdx.x < end-start){
        xshared[threadIdx.x] = x_sorted[start+threadIdx.x];
        cshared[threadIdx.x] = c_sorted[start+threadIdx.x];
      }
      __syncthreads();
      for(j=0; j<end-start; j++){
        double dis = abs(xshared[j]-i);
        if( dis < ns/2.0){
           fw[2*i]++;
           fw[2*i+1]++;
           //double kervalue = evaluate_kernel(dis, es_c, es_beta);
           //fw[i]  = cuCadd (fw[i], make_cuDoubleComplex(cuCreal(cshared[j])*kervalue, cuCimag(cshared[j])*kervalue));
        }
      }
    }
  }
}

#if 0
int main(void)
{
  // Parameter setting
  int N1 = 32;
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
  cudaFree(x);
  cudaFree(c);
  cudaFree(csorted);
  cudaFree(fw);
  free(h_binsize); 
  return 0;
}
#endif
