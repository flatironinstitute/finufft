#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "utils.h"

using namespace std;

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

static __forceinline__ __device__
FLT evaluate_kernel(FLT x, FLT es_c, FLT es_beta)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 2/17/17 */
{   
  //return exp(es_beta * (sqrt(1.0 - es_c*x*x) - 1));
  //return x;
  return 1.0;
}

static __inline__ __device__
void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w, const double upsampfac)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
  FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
  // insert the auto-generated code which expects z, w args, writes to ker...
  if (upsampfac==2.0) {     // floating point equality is fine here
#include "ker_horner_allw_loop.c"
  }
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
    //x_rescaled = RESCALE(x[i],nf1,1);
    //y_rescaled = RESCALE(y[i],nf2,1);
    x_rescaled=x[i];
    y_rescaled=y[i];
    binx = floor(x_rescaled/bin_size_x)+1;
    biny = floor(y_rescaled/bin_size_y)+1;
    binidx = binx+biny*nbinx;
    oldidx = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

__global__
void FillGhostBin_2d(int nbinx, int nbiny, int*bin_size)
{
  int ix = blockDim.x*blockIdx.x + threadIdx.x;
  int iy = blockDim.y*blockIdx.y + threadIdx.y;
  if ( ix < nbinx && iy < nbiny){
    if(iy == 0)
      bin_size[ix+iy*nbinx] = bin_size[ix+(nbiny-2)*nbinx];
    if(iy == nbiny-1)
      bin_size[ix+iy*nbinx] = bin_size[ix+1*nbinx];
    if(ix == 0)
      bin_size[ix+iy*nbinx] = bin_size[(nbinx-2)+iy*nbinx];
    if(ix == nbinx-1)
      bin_size[ix+iy*nbinx] = bin_size[1+iy*nbinx];
    if(ix == 0 && iy == 0)
      bin_size[ix+iy*nbinx] = bin_size[(nbinx-2)+(nbiny-2)*nbinx];
    if(ix == 0 && iy == nbiny-1)
      bin_size[ix+iy*nbinx] = bin_size[(nbinx-2)+1*nbinx];
    if(ix == nbinx-1 && iy == 0)
      bin_size[ix+iy*nbinx] = bin_size[1+(nbiny-2)*nbinx];
    if(ix == nbinx-1 && iy == nbiny-1)
      bin_size[ix+iy*nbinx] = bin_size[1+1*nbinx];
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
void prescan(int n, int* bin_size, int* bin_startpts, int* scanblock_sum)
// only works for n is power of 2
{
  __shared__ int temp[max_shared_mem];
  int thid=threadIdx.x;
  int offset=1;
  int nelem=2*blockDim.x;

  if(2*thid+1<n){
    temp[2*thid+1]=bin_size[2*thid+1];
  }else{
    temp[2*thid+1]=0;
  }
  if(2*thid<n){
    temp[2*thid]=bin_size[2*thid];
  }else{
    temp[2*thid]=0;
  }

  for(int d = nelem>>1; d>0; d>>=1)
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

  if(thid==0) {temp[nelem-1]=0;}

  for(int d=1; d<nelem; d*=2)
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
  
  if(2*thid+1<n){
    bin_startpts[2*thid+1]=temp[2*thid+1];
  }
  if(2*thid<n){
    bin_startpts[2*thid]=temp[2*thid];
  }
  *scanblock_sum=temp[n-1]+bin_size[n-1];
}

__global__
void uniformUpdate(int n, int* data, int* buffer)
{
  __shared__ int buf;
  int pos=blockIdx.x*blockDim.x+threadIdx.x;
  if( threadIdx.x ==0){
    buf=buffer[blockIdx.x];
  }
  __syncthreads();
  if(pos<n)
    data[pos] += buf;
  if(pos==0)
    data[n] = buffer[gridDim.x];
}

__global__
void PtsRearrage_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx,
                    int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *x_sorted,
                    FLT *y, FLT *y_sorted, gpuComplex *c, gpuComplex *c_sorted)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int binx, biny;
  int binidx;
  FLT x_rescaled, y_rescaled;
  if( i < M){
    //x_rescaled = RESCALE(x[i],nf1,1);
    //y_rescaled = RESCALE(y[i],nf2,1);
    x_rescaled=x[i];
    y_rescaled=y[i];
    binx = floor(x_rescaled/bin_size_x)+1;
    biny = floor(y_rescaled/bin_size_y)+1;
    binidx = binx+biny*nbinx;

    x_sorted[bin_startpts[binidx]+sortidx[i]] = x_rescaled;
    y_sorted[bin_startpts[binidx]+sortidx[i]] = y_rescaled;
    c_sorted[bin_startpts[binidx]+sortidx[i]] = c[i];

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
    c_sorted[ bin_startpts[binidx]+sortidx[i] ] = c[i];
  }
}

__global__
void Spread_2d_Odriven(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, 
                       int *bin_startpts, FLT *x_sorted, FLT *y_sorted, 
                       gpuComplex *c_sorted, gpuComplex *fw, int ns,
                       int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width)
{
  __shared__ FLT xshared[max_shared_mem/4];
  __shared__ FLT yshared[max_shared_mem/4];
  __shared__ gpuComplex cshared[max_shared_mem/4];

  int ix = blockDim.x*blockIdx.x+threadIdx.x;// output index, coord of the index
  int iy = blockDim.y*blockIdx.y+threadIdx.y;// output index, coord of the index
  int outidx = ix + iy*fw_width;
  int binxLo = blockIdx.x*nbin_block_x;
  int binxHi = binxLo+nbin_block_x+1 < nbinx-1 ? binxLo+nbin_block_x+1 : nbinx-1;
  int binyLo = blockIdx.y*nbin_block_y;
  int binyHi = binyLo+nbin_block_y+1 < nbiny-1 ? binyLo+nbin_block_y+1 : nbiny-1;
  int start, end, j, bx, by, bin;
  FLT disx, disy, kervalue1, kervalue2;
  //FLT tr=0.0, ti=0.0;
  gpuComplex t=make_cuDoubleComplex(0,0);
  // run through all bins
  for(by=binyLo; by<=binyHi; by++){
    //for(bx=binxLo; bx<=binxHi; bx++){
      //bin = bx+by*nbinx;
      //start = bin_startpts[bin];
      //end   = bin_startpts[bin+1];
      start = bin_startpts[binxLo+by*nbinx];
      end   = bin_startpts[binxHi+by*nbinx+1];
      for(int tid=threadIdx.x+blockDim.x*threadIdx.y; tid<end-start; tid+=blockDim.x*blockDim.y){
        xshared[tid] = x_sorted[start+tid];
        yshared[tid] = y_sorted[start+tid];
        cshared[tid] = c_sorted[start+tid];
      }
      __syncthreads();
      if( ix < nf1 && iy < nf2){
        for(j=0; j<end-start; j++){
          disx = abs(xshared[j]-ix);
          disy = abs(yshared[j]-iy);
          gpuComplex c=cshared[j];
          if( (disx < 7.0/2.0) && (disy < 7.0/2.0)){
            kervalue1 = evaluate_kernel(disx, es_c, es_beta);
            kervalue2 = evaluate_kernel(disy, es_c, es_beta);
            t.x+=kervalue1*kervalue2;
            t.y+=kervalue1*kervalue2;
            //t.x++;
            //t.y++;
            //t = cuCadd(t, make_cuDoubleComplex(c.x*kervalue, c.y*kervalue));
            //tr += cshared[2*j]*kervalue;
            //ti += cshared[2*j+1]*kervalue;
          }
        }
      }
    //}
  } 
  if( ix < nf1 && iy < nf2){
    fw[outidx]=t;
  }
    //fw[2*outidx]   = tr;
    //fw[2*outidx+1] = ti;
}

__global__
void Spread_2d_Idriven(FLT *x, FLT *y, gpuComplex *c, gpuComplex *fw, int M, const int ns,
                       int nf1, int nf2, FLT es_c, FLT es_beta, int fw_width)
{
  int xstart,ystart,xend,yend;
  int xx, yy, ix, iy;
  int outidx;
  //FLT ker1[7];
  //FLT ker2[7];
  //FLT ker1val, ker2val;
  //double sigma=2.0;

  FLT x_rescaled, y_rescaled;
  for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
    //x_rescaled = RESCALE(x[i],nf1,1);
    //y_rescaled = RESCALE(y[i],nf2,1);
    x_rescaled=x[i];
    y_rescaled=y[i];
    xstart = ceil(x_rescaled - ns/2.0);
    ystart = ceil(y_rescaled - ns/2.0);
    xend = floor(x_rescaled + ns/2.0);
    yend = floor(y_rescaled + ns/2.0);

#if 0
    FLT x1=(FLT)xstart-x_rescaled;
    FLT y1=(FLT)ystart-y_rescaled;
    eval_kernel_vec_Horner(ker1,x1,ns,sigma);
    eval_kernel_vec_Horner(ker2,y1,ns,sigma);
#endif
    for(yy=ystart; yy<=yend; yy++){
#if 0
       ker2val=ker2[yy-ystart];
#endif
       for(xx=xstart; xx<=xend; xx++){
#if 0
          ker1val=ker1[xx-xstart];
#endif
          ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
          iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
          outidx = ix+iy*fw_width;
#if 0
          FLT kervalue=ker1val*ker2val;
#endif
          FLT disx=abs(x_rescaled-xx);
          FLT disy=abs(y_rescaled-yy);
          FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta);
          FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta);
          atomicAdd(&fw[outidx].x, kervalue1*kervalue2);
          atomicAdd(&fw[outidx].y, kervalue1*kervalue2);
          //atomicAdd(&fw[2*outidx  ], 1.0);
          //atomicAdd(&fw[2*outidx+1], 1.0);
       }
    }

  }

}

__global__
void CreateSortIdx (int M, int nf1, int nf2, FLT *x, FLT *y, int* sortidx)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  FLT x_rescaled,y_rescaled;
  if (i < M){
    //x_rescaled = RESCALE(x[i],nf1,1);
    //y_rescaled = RESCALE(y[i],nf2,1);
    x_rescaled=x[i];
    y_rescaled=y[i];
    sortidx[i] = floor(x_rescaled) + floor(y_rescaled)*nf1;
  }
}
