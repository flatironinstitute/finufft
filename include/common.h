#ifdef T

#include <fftw_defs.h>
#include <nufft_opts.h>

// defs internal to common.cpp...
typedef std::complex<double> dcomplex;
// (slightly sneaky since name duplicated by mwrap - unrelated)

// common.cpp provides...
int TEMPLATE(setup_spreader_for_nufft,T)(TEMPLATE(spread_opts,T) &spopts, T eps, nufft_opts opts);
void TEMPLATE(set_nf_type12,T)(BIGINT ms, nufft_opts opts, TEMPLATE(spread_opts,T) spopts,BIGINT *nf);
void TEMPLATE(set_nhg_type3,T)(T S, T X, nufft_opts opts, TEMPLATE(spread_opts,T) spopts,
		  BIGINT *nf, T *h, T *gam);
void TEMPLATE(onedim_dct_kernel,T)(BIGINT nf, T *fwkerhalf, TEMPLATE(spread_opts,T) opts);
void TEMPLATE(onedim_fseries_kernel,T)(BIGINT nf, T *fwkerhalf, TEMPLATE(spread_opts,T) opts);
void TEMPLATE(onedim_nuft_kernel,T)(BIGINT nk, T *k, T *phihat, TEMPLATE(spread_opts,T) opts);
void TEMPLATE(deconvolveshuffle1d,T)(int dir,T prefac,T* ker,BIGINT ms,T *fk,
			 BIGINT nf1,TEMPLATE(FFTW_CPX,T)* fw,int modeord);
void TEMPLATE(deconvolveshuffle2d,T)(int dir,T prefac,T *ker1, T *ker2,
			 BIGINT ms,BIGINT mt,
			 T *fk, BIGINT nf1, BIGINT nf2, TEMPLATE(FFTW_CPX,T)* fw,
			 int modeord);
void TEMPLATE(deconvolveshuffle3d,T)(int dir,T prefac,T *ker1, T *ker2,
			 T *ker3, BIGINT ms, BIGINT mt, BIGINT mu,
			 T *fk, BIGINT nf1, BIGINT nf2, BIGINT nf3,
			 TEMPLATE(FFTW_CPX,T)* fw, int modeord);
#endif //def T

