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

#if 0
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
#endif

__global__
void CalcBinSize_2d(int M, int nf1, int nf2, int  bin_size_x, int bin_size_y, int nbinx,
                    int nbiny, int* bin_size, double *x, double *y, int* sortidx)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int binidx, binx, biny;
  int oldidx;
  double x_rescaled,y_rescaled;
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
  if ( ix < nbinx & iy < nbiny){
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
  extern __shared__ unsigned int temp[];
  int i = threadIdx.x;
  //temp[i] = (i > 0) ? bin_size[i-1] : 0;
  temp[i] = bin_size[i];
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

__global__
void PtsRearrage_2d(int M, int nf1, int nf2, int bin_size_x, int bin_size_y, int nbinx, int nbiny,
                    int* bin_startpts, int* sortidx, double *x, double *x_sorted, 
                    double *y, double *y_sorted, double *c, double *c_sorted)
{
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int binx, biny;
  int binidx;
  double x_rescaled, y_rescaled;
  if( i < M){
    x_rescaled = RESCALE(x[i],nf1,1);
    y_rescaled = RESCALE(y[i],nf2,1);
    binx = floor(x_rescaled/bin_size_x)+1;
    biny = floor(y_rescaled/bin_size_y)+1;
    binidx = binx+biny*nbinx;
   
    //binidx = floor(x_rescaled/bin_size_x);
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
#if 1
__global__
void Spread_2d(int nbin_block_x, int nbin_block_y, int nbinx, int nbiny, int *bin_startpts,
               double *x_sorted, double *y_sorted, double *c_sorted, double *fw, int ns, 
               int nf1, int nf2, double es_c, double es_beta)
{
  __shared__ double xshared[max_shared_mem/4];
  __shared__ double yshared[max_shared_mem/4];
  __shared__ double cshared[2*max_shared_mem/4];

  int ix = blockDim.x*blockIdx.x+threadIdx.x;// output index, coord of the index
  int iy = blockDim.y*blockIdx.y+threadIdx.y;// output index, coord of the index
  int outidx = ix + iy*nf1;
  int binxLo = blockIdx.x*nbin_block_x;
  int binxHi = binxLo+nbin_block_x+1;
  int binyLo = blockIdx.y*nbin_block_y;
  int binyHi = binyLo+nbin_block_y+1;
  int start, end, j, bx, by, bin;
  // run through all bins
  if( ix < nf1 && iy < nf2){
    for(by=binyLo; by<=binyHi; by++){
      for(bx=binxLo; bx<=binxHi; bx++){
        bin = bx+by*nbinx;
        start = bin_startpts[bin];
        end   = bin_startpts[bin+1];
        if( outidx < end-start){
          xshared[outidx] = x_sorted[start+outidx];
          yshared[outidx] = y_sorted[start+outidx];
          cshared[outidx] = c_sorted[start+outidx];
        }
        __syncthreads();
        for(j=0; j<end-start; j++){
          double disx = abs(xshared[j]-ix);
          double disy = abs(yshared[j]-iy);
          if( disx < ns/2.0 && disy < ns/2.0){
             fw[2*outidx] ++;
             fw[2*outidx+1] ++;
             //double kervalue = evaluate_kernel(dis, es_c, es_beta);
             //fw[i]  = cuCadd (fw[i], make_cuDoubleComplex(cuCreal(cshared[j])*kervalue, cuCimag(cshared[j])*kervalue));
          }
        }
      }
    }
  }
}
#endif
