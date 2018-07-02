#include <iostream>
#include <math.h>
#include <helper_cuda.h>

using namespace std;

#define PI (double)M_PI
#define M_1_2PI 0.159154943091895336
#define RESCALE(x,N,p) (p ? \
             ((x*M_1_2PI + (x<-PI ? 1.5 : (x>PI ? -0.5 : 0.5)))*N) : \
             (x<0 ? x+N : (x>N ? x-N : x)))
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
void CalcBinSize_1d(int M, int nf1, int  bin_size_x, int nbinx,
                    int* bin_size, double *x, int* sortidx)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int binidx, binx;
  int oldidx;
  double x_rescaled;
  if (i < M){
    x_rescaled = RESCALE(x[i],nf1,1);
    binx = floor(x_rescaled/bin_size_x)+1;
    binidx = binx;
    oldidx = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

__global__
void FillGhostBin_1d(int bin_size_x, int nbinx, int*bin_size)
{
  int ix = blockDim.x*blockIdx.x + threadIdx.x;
  if ( ix < nbinx ){
    if(ix == 0)
      bin_size[ix] = bin_size[(nbinx-2)];
    if(ix == nbinx-1)
      bin_size[ix] = bin_size[1];
  }
}

// An exclusive scan of bin_size, only works for 1 block (!) 
__global__
void BinsStartPts_1d(int M, int totalnumbins, int* bin_size, int* bin_startpts)
{
  __shared__ int temp[max_shared_mem];
  int i = threadIdx.x;
  //temp[i] = (i > 0) ? bin_size[i-1] : 0;
  if ( i < totalnumbins){
    temp[i] = (i<totalnumbins) ? bin_size[i]:0;
    __syncthreads();
    for(int offset = 1; offset < totalnumbins; offset*=2){
      if( i >= offset)
        temp[i] += temp[i - offset];
      else
        temp[i] = temp[i];
      __syncthreads();
    }
    bin_startpts[i+1] = temp[i];
    if(i == 0)
      bin_startpts[i] = 0;
  }
}

__global__
void PtsRearrage_1d(int M, int nf1, int bin_size_x, int nbinx,
                    int* bin_startpts, int* sortidx, double *x, double *x_sorted, 
                    double *c, double *c_sorted)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int binx;
  int binidx;
  double x_rescaled;
  if( i < M){
    x_rescaled = RESCALE(x[i],nf1,1);
    binx = floor(x_rescaled/bin_size_x)+1;
    binidx = binx;
   
    x_sorted[bin_startpts[binidx]+sortidx[i]]       = x_rescaled;
    
    if( binx == 1 ){
      binidx = (nbinx-1);
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled + nf1;
    }
    if( binx == nbinx-2 ){
      binidx = 0;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled - nf1;
    }
    c_sorted[ 2*(bin_startpts[binidx]+sortidx[i]) ] = c[2*i];
    c_sorted[ 2*(bin_startpts[binidx]+sortidx[i])+1 ] = c[2*i+1];
  }
}

__global__
void Spread_1d(int nbin_block_x, int nbinx, int *bin_startpts,
               double *x_sorted, double *c_sorted, double *fw, int ns, 
               int nf1, double es_c, double es_beta)
{
  __shared__ double xshared[max_shared_mem/4];
  __shared__ double cshared[2*max_shared_mem/4];

  int ix = blockDim.x*blockIdx.x+threadIdx.x;// output index, coord of the index
  int outidx = ix;
  int tid = threadIdx.x;
  int binxLo = blockIdx.x*nbin_block_x;
  int binxHi = binxLo+nbin_block_x+1;
  int start, end, j, bx, bin;
  // run through all bins
  if( ix < nf1 ){
      for(bx=binxLo; bx<=binxHi; bx++){
        bin = bx;
        start = bin_startpts[bin];
        end   = bin_startpts[bin+1];
        if( tid < end-start){
          xshared[tid] = x_sorted[start+tid];
          cshared[2*tid]   = c_sorted[2*(start+tid)];
          cshared[2*tid+1] = c_sorted[2*(start+tid)+1];
        }
        __syncthreads();
        for(j=0; j<end-start; j++){
          double disx = abs(xshared[j]-ix);
          if( disx < ns/2.0 ){
             fw[2*outidx] ++;
             fw[2*outidx+1] ++;
             //double kervalue = evaluate_kernel(disx, es_c, es_beta);
             //fw[2*outidx]   += cshared[2*j]*kervalue;
             //fw[2*outidx+1] += cshared[2*j+1]*kervalue;
          }
        }
      }
  }
}
