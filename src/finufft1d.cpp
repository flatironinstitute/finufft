#include "finufft1d.h"

#include <iostream>
#include <iomanip>
using namespace std;

int finufft1d1(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk)
{
 /*  Type-1 1D complex nonuniform FFT.

               1 nj-1
     fk(k1) = -- SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
              nj j=0                            
   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj     location of sources on interval [-pi,pi].
     cj     complex source strengths, interleaving Re & Im parts (2*nj doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
   Outputs:
     fk     complex Fourier transform values (size ms, increasing mode ordering)
            stored as alternating Re & Im parts (2*ms doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        corresponding coefficient from the kernel alone.
     The latter kernel FFT is precomputed in what is called step 0 in the code.

   Written in real-valued C style (for speed) & FFTW arrays. Barnett 1/22/17
 */
  
  int debug = 1;      // should be an input opt
  spread_opts spopts;
  int ier_set = set_KB_opts_from_eps(spopts,eps);
  double params[4];
  ier_set = get_kernel_params_for_eps(params,eps);
  int nspread = params[1];
  double R = 2.0;              // upsampling - todo check it
  BIGINT nf1 = 2*(BIGINT)(0.5*R*ms);  // is even.  use a next235 ?
  if (nf1<2*nspread) nf1=2*nspread;  // otherwise spread fails
  int dir = 1;        // spread
  cout << scientific << setprecision(15);  // for debug

  if (debug) printf("d1d: ms=%d nf1=%d nj=%d ...\n",ms,nf1,nj); 
  CNTime timer; timer.start();
  // STEP 0: get DCT of half of spreading kernel, since it's real symmetric
  double *fwkerhalf = fftw_alloc_real(nf1/2+1);
  double prefac_unused_dims;
  onedim_dct_kernel(nf1, fwkerhalf, prefac_unused_dims, spopts);
  double t=timer.elapsedsec();
  if (debug) printf("onedim_dct_kernel:\t %.3g s\n", t);
  //for (int j=0;j<=nf1/2;++j) cout<<fwkerhalf[j]<<endl;

  int nth = omp_get_max_threads();     // set up any multithreaded fftw stuff
  #ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
  #endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1);    // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_1d(nf1,fw,fw,fftsign, FFTW_ESTIMATE);
  if (debug) printf("fft plan\t\t %.3g s\n", timer.elapsedsec());

  timer.restart();
  // Step 1: spread from irregular points to regular grid
  int ier_spread = twopispread1d(nf1,(double*)fw,nj,xj,cj,dir,params);
  if (debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  timer.restart();
  // Step 2:  Call FFT
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (debug) printf("fft (%d thr):\t\t %.3g s\n", nth, timer.elapsedsec());
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;
  //for (int j=0;j<=nf1/2;++j) cout<<fwkerhalf[j]<<endl;

  timer.restart();
  // Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  double prefac = 1.0/(prefac_unused_dims*prefac_unused_dims*nj);
  //  deconvolveshuffle

  BIGINT k0 = ms/2;    // index shift in output freqs  *** EVEN ONLY

  for (BIGINT k=0;k<=(ms-1)/2;++k) {               // non-neg freqs k
    //cout<< k0+k<<"\t"<<k<<endl;
    fk[2*(k0+k)] = prefac * fw[k][0] / fwkerhalf[k];          // re
    fk[2*(k0+k)+1] = prefac * fw[k][1] / fwkerhalf[k];        // im
  }
  for (BIGINT k=-1;k>=-ms/2;--k) {                 // neg freqs k
    //cout<< k0+k<<"\t"<<nf1+k<<"\t"<<k<<endl;
    fk[2*(k0+k)] = prefac * fw[nf1+k][0] / fwkerhalf[-k];     // re
    fk[2*(k0+k)+1] = prefac * fw[nf1+k][1] / fwkerhalf[-k];   // im
  }
  if (debug) printf("deconvolve & copy out:\t %.3g s\n", timer.elapsedsec());
  //for (int j=0;j<ms;++j) cout<<fk[2*j]<<"\t"<<fk[2*j+1]<<endl;

  fftw_free(fw);
  fftw_free(fwkerhalf);
  if (debug>1) printf("freed\n");
  return ier_spread;
}


int finufft1d2(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk)
{
 /*  Type-2 1D complex nonuniform FFT.

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1 
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj     location of sources on interval [-pi,pi].
     fk     complex Fourier transform values (size ms, increasing mode ordering)
            stored as alternating Re & Im parts (2*ms doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
   Outputs:
     cj     complex source strengths, interleaving Re & Im parts (2*nj doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written in real-valued C style (for speed) & FFTW arrays. Barnett 1/25/17
 */
  


  return 0;
}
