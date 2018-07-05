#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "utils.h"

using namespace std;

#define PI (FLT)M_PI
#define M_1_2PI 0.159154943091895336
#define RESCALE(x,N,p) (p ? \
             ((x*M_1_2PI + (x<-PI ? 1.5 : (x>PI ? -0.5 : 0.5)))*N) : \
             (x<0 ? x+N : (x>N ? x-N : x)))
#define max_shared_mem 6000

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__
FLT evaluate_kernel(FLT x, FLT es_c, FLT es_beta)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{
    return exp(es_beta * (sqrt(1.0 - es_c*x*x) - 1));
}

__global__
void CalcBinSize_2d(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, int nbinx,
                    int nbiny, int* bin_size, FLT *x, FLT *y, int* sortidx)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int binidx, binx, biny;
  int oldidx;
  FLT x_rescaled,y_rescaled;
  if (i < M){
    x_rescaled = RESCALE(x[i],nf1,1);
    y_rescaled = RESCALE(y[i],nf2,1);
    binx = floor(x_rescaled/bin_size_x)+1;
    biny = floor(y_rescaled/bin_size_y)+1;
    binidx = binx+biny*nbinx;
    oldidx = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

__global__
void FillGhostBin_2d(int bin_size_x, int bin_size_y, int nbinx, int nbiny, int*bin_size)
{
  int ix = blockDim.x*blockIdx.x + threadIdx.x;
  int iy = blockDim.y*blockIdx.y + threadIdx.y;
  if ( ix < nbinx && iy < nbiny){
    if( iy == 0 )
      bin_size[ix+iy*nbiny] = bin_size[ix+(nbiny-2)*nbiny];
    if(iy == nbiny-1)
      bin_size[ix+iy*nbiny] = bin_size[ix+1*nbiny];
    __syncthreads();
    if(ix == 0)
      bin_size[ix+iy*nbiny] = bin_size[(nbinx-2)+iy*nbiny];
    if(ix == nbinx-1)
      bin_size[ix+iy*nbiny] = bin_size[1+iy*nbiny];
  }
}

// An exclusive scan of bin_size, only works for 1 block (!) 
__global__
void BinsStartPts_2d(int M, int totalnumbins, int* bin_size, int* bin_startpts)
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
void prescan(int n, int* bin_size, int* bin_startpts)
// only works for n is power of 2
{
  extern __shared__ int temp[];
  int thid=threadIdx.x;
  int offset=1;

  temp[2*thid]=bin_size[2*thid];
  temp[2*thid+1]=bin_size[2*thid+1];

  for(int d = n>>1; d>0; d>>=1)
  {
    __syncthreads();
    if(thid<d)
    {
      int ai=offset*(2*thid+1)-1;
      int bi=offset*(2*thid+2)-1;
      temp[bi]+=temp[ai];
    }
    offset*=2;
  }
  if(thid==0) {temp[n-1]=0;}
  for(int d=1; d<n; d*=2)
  {
    offset>>=1;
    __syncthreads();
    if(thid<d)
    {
      int ai=offset*(2*thid+1)-1;
      int bi=offset*(2*thid+2)-1;

      int t=temp[ai];
      temp[ai]=temp[bi];
      temp[bi]+=t;
    }
  }
  __syncthreads();
  bin_startpts[2*thid]=temp[2*thid];
  bin_startpts[2*thid+1]=temp[2*thid+1];

}

__global__
void PtsRearrage_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                    int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted, 
                    FLT *y, FLT *y_sorted, FLT *c, FLT *c_sorted)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int binx, biny;
  int binidx;
  FLT x_rescaled, y_rescaled;
  if( i < M){
    x_rescaled = RESCALE(x[i],nf1,1);
    y_rescaled = RESCALE(y[i],nf2,1);
    binx = floor(x_rescaled/bin_size_x)+1;
    biny = floor(y_rescaled/bin_size_y)+1;
    binidx = binx+biny*nbinx;
   
    //binidx = floor(x_rescaled/bin_size_x);
    x_sorted[bin_startpts[binidx]+sortidx[i]] = x_rescaled;
    y_sorted[bin_startpts[binidx]+sortidx[i]] = y_rescaled;
    c_sorted[2*(bin_startpts[binidx]+sortidx[i])]   = c[2*i];
    c_sorted[2*(bin_startpts[binidx]+sortidx[i])+1] = c[2*i+1];
    
    // four edges
    if( binx == 1 ){
      binidx = (nbinx-1)+biny*nbinx;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled + nf1;
      y_sorted[ bin_startpts[binidx]+sortidx[i] ] = y_rescaled;
    }
    if( binx == nbinx-2 ){
      binidx = biny*nbinx;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled - nf1;
      y_sorted[ bin_startpts[binidx]+sortidx[i] ] = y_rescaled;
    }
    if( biny == 1 ){
      binidx = binx+(nbiny-1)*nbinx;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled;
      y_sorted[ bin_startpts[binidx]+sortidx[i] ] = y_rescaled + nf2;
    }
    if( biny == nbiny-2 ){
      binidx = binx;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled;
      y_sorted[ bin_startpts[binidx]+sortidx[i] ] = y_rescaled - nf2;
    }
    // four corner
    if( binx == 1 && biny == 1){
      binidx = (nbinx-1) + (nbiny-1)*nbinx;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled + nf1;
      y_sorted[ bin_startpts[binidx]+sortidx[i] ] = y_rescaled + nf2;
    }
    if( binx == 1 && biny == nbiny-2){
      binidx = nbinx-1;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled + nf1;
      y_sorted[ bin_startpts[binidx]+sortidx[i] ] = y_rescaled - nf2;
    }
    if( binx == nbinx-2 && biny == 1){
      binidx = (nbiny-1)*nbinx;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled - nf1;
      y_sorted[ bin_startpts[binidx]+sortidx[i] ] = y_rescaled + nf2;
    }
    if( binx == nbinx-2 && biny == nbiny-2){
      binidx = 0;
      x_sorted[ bin_startpts[binidx]+sortidx[i] ] = x_rescaled - nf1;
      y_sorted[ bin_startpts[binidx]+sortidx[i] ] = y_rescaled - nf2;
    }
    c_sorted[ 2*(bin_startpts[binidx]+sortidx[i]) ] = c[2*i];
    c_sorted[ 2*(bin_startpts[binidx]+sortidx[i])+1 ] = c[2*i+1];
  }
}
#if 1
__global__
void Spread_2d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, int *bin_startpts,
                       FLT *x_sorted, FLT *y_sorted, FLT *c_sorted, FLT *fw, int ns, 
                       int nf1, int nf2, FLT es_c, FLT es_beta)
{
  __shared__ FLT xshared[max_shared_mem/4];
  __shared__ FLT yshared[max_shared_mem/4];
  __shared__ FLT cshared[2*max_shared_mem/4];

  int ix = blockDim.x*blockIdx.x+threadIdx.x;// output index, coord of the index
  int iy = blockDim.y*blockIdx.y+threadIdx.y;// output index, coord of the index
  int outidx = ix + iy*nf1;
  int tid = threadIdx.x + blockDim.x*threadIdx.y;
  int binxLo = blockIdx.x*nbin_block_x;
  int binxHi = binxLo+nbin_block_x+1;
  int binyLo = blockIdx.y*nbin_block_y;
  int binyHi = binyLo+nbin_block_y+1;
  int start, end, j, bx, by, bin;
  FLT tr=0.0, ti=0.0;
  // run through all bins
  if( ix < nf1 && iy < nf2){
    for(by=binyLo; by<=binyHi; by++){
      //for(bx=binxLo; bx<=binxHi; bx++){
        bin = bx+by*nbinx;
        start = bin_startpts[binxLo+by*nbinx];
        end   = bin_startpts[binxHi+by*nbinx+1];
        if( tid < end-start){
          xshared[tid] = x_sorted[start+tid];
          yshared[tid] = y_sorted[start+tid];
          cshared[2*tid]   = c_sorted[2*(start+tid)];
          cshared[2*tid+1] = c_sorted[2*(start+tid)+1];
        }
        __syncthreads();
        for(j=0; j<end-start; j++){
          FLT disx = abs(xshared[j]-ix);
          FLT disy = abs(yshared[j]-iy);
          if( disx < ns/2.0 && disy < ns/2.0){
             tr++;
             ti++;
             //FLT kervalue = evaluate_kernel(sqrt(disx*disx+disy*disy), es_c, es_beta);
             //tr += cshared[2*j]*kervalue;
             //ti += cshared[2*j+1]*kervalue;
          }
        }
      //}
    }
    fw[2*outidx]   = tr;
    fw[2*outidx+1] = ti;
  }
}
#if 1
__global__
void Spread_2d_Idriven(FLT *x, FLT *y, FLT *c, FLT *fw, int M, int ns, 
                       int nf1, int nf2, FLT es_c, FLT es_beta)
{
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int xstart, ystart;
  int xx, yy, ix, iy;
  int outidx;
  FLT x_rescaled, y_rescaled;
  if( i<M ){
    x_rescaled = RESCALE(x[i],nf1,1);
    y_rescaled = RESCALE(y[i],nf2,1);
    xstart = ceil(x_rescaled - ns/2.0);
    ystart = ceil(y_rescaled - ns/2.0);
    for(yy=ystart; yy<ystart+ns; yy++){
       for(xx=xstart; xx<xstart+ns; xx++){
          ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
          iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
          outidx = ix+iy*nf1;
          //FLT disx=abs(x_sorted[i]- (xstart+dx));
          //FLT disy=abs(y_sorted[i]- (ystart+dy));
          //FLT kervalue = evaluate_kernel(sqrt(disx*disx+disy*disy), es_c, es_beta);
          atomicAdd((double*) &fw[2*outidx  ], 1.0);
          atomicAdd((double*) &fw[2*outidx+1], 1.0);
       }
    }
    
  }

}
#endif
#endif
