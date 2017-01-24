#include "finufft1d.h"

#include <iostream>
#include <iomanip>
using namespace std;

int finufft1d1(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk)
{
 /*  Type-1 1D NUFFT.

     if (iflag>0) then

               1  nj
     fk(k1) = -- SUM cj(j) exp(+i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
              nj j=1                            

     else
               1  nj
     fk(k1) = -- SUM cj(j) exp(-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
              nj j=1                            

   Inputs:
     nj     number of sources (integer of type BIGINT; see cnufftspread.h)
     xj     location of sources on interval [-pi,pi].
     cj     strengths of sources (complex *16)
     iflag  determines sign of FFT (see above)
     eps    precision requested
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
   Outputs:
     fk     Fourier transform values (size ms, mode numbers as above).
            Complex stored as alternating real and imag double reals.
     returned value - error return code, as returned by cnufftspread:
                      0 indicates success.

     The type 1 NUFFT proceeds in three main steps (see [GL]).

     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        corresponding coefficient from the kernel alone.
     The latter kernel FFT is precomputed in what is called step 0 in the code.

   Written in real-valued C style (for speed) & FFTW arrays. Barnett 1/22/17
 */
  
  //spread_opts spopts;
  //int ier_set = set_KB_opts_from_eps(spopts,eps);
  int debug = 1;
  double params[4];
  int ier_set = get_kernel_params_for_eps(params,eps);
  int nspread = params[1];
  BIGINT nf1 = 2*ms;  // adjust, and use a next235, ensure even
  int dir = 1;        // spread
  cout << scientific << setprecision(15);

  // STEP 0: get DCT of half of spreading kernel, since it's real symmetric
  double xker[1] = {0.0};           // to get kernel, place source at origin
  double cker[2] = {1.0*nj,0.0};    // complex, strength nj for norm definition
  double *fwker = (double *)malloc(sizeof(double)*2*nf1);  // complex ker eval
  twopispread1d(nf1,fwker,1,xker,cker,dir,params);        // fill complex ker
  cout<<"ker spread done\n";  // *** todo insert debug timers
  BIGINT nhalf = nf1/2+1;
  printf("ms=%d nf1=%d nj=%d nhalf=%d nspread=%d\n",ms,nf1,nj,nhalf,nspread);
  double *kerhalf = fftw_alloc_real(nhalf);
  // note: in-place, and we make plan before filling input array...
  fftw_plan p = fftw_plan_r2r_1d(nhalf,kerhalf,kerhalf,FFTW_REDFT00,
				 FFTW_ESTIMATE);  // note no fftsign
  for (BIGINT i=0; i<nhalf; ++i)  kerhalf[i] = 0.0;   // zero it
  for (BIGINT i=nhalf-1-nspread/2; i<nhalf; ++i)
    kerhalf[i] = fwker[2*i];    // copy Re kernel to end real fft input array
  //for (int j=0;j<nhalf;++j) cout<<kerhalf[j]<<endl;
  free(fwker);
  //printf("kerhalf filled\n");
  fftw_execute(p);
  fftw_destroy_plan(p);
  cout<<"ker fft done\n";
  //for (int j=0;j<nhalf;++j) cout<<kerhalf[j]<<endl;
  //printf("fftw kerhalf done\n");

  fftw_complex *fw = fftw_alloc_complex(nf1);    // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  p = fftw_plan_dft_1d(nf1,fw,fw,fftsign, FFTW_ESTIMATE);

  // Step 1: spread from irregular points to regular grid
  int ier_spread = twopispread1d(nf1,(double*)fw,nj,xj,cj,dir,params);

  cout<<"spread done\n";
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  fftw_execute(p);
  fftw_destroy_plan(p);

  cout<<"fft done\n"; //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<fw[j][1]<<endl;

  // Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  BIGINT k0 = ms/2;    // index shift in output freqs
  for (BIGINT k=0;k<=(ms-1)/2;++k) {               // non-neg freqs k
    fk[2*(k0+k)] = fw[k][0] / kerhalf[k];          // re
    fk[2*(k0+k)+1] = fw[k][1] / kerhalf[k];        // im
  }
  for (BIGINT k=-1;k>=-ms/2;--k) {                 // neg freqs k
    fk[2*(k0+k)] = fw[nf1+k][0] / kerhalf[-k];     // re
    fk[2*(k0+k)+1] = fw[nf1+k][1] / kerhalf[-k];   // im
  }
  //for (int j=0;j<ms;++j) cout<<fk[j]<<endl;

  fftw_free(fw);
  fftw_free(kerhalf);
  return ier_spread;
}
