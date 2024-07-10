#include <algorithm>
#include <finufft/fft.h>

using namespace std;

std::vector<int> gridsize_for_fft(FINUFFT_PLAN p) {
  // local helper func returns a new int array of length dim, extracted from
  // the finufft plan, that fftw_plan_many_dft needs as its 2nd argument.
  if (p->dim == 1) return {(int)p->nf1};
  if (p->dim == 2) return {(int)p->nf2, (int)p->nf1};
  // if (p->dim == 3)
  return {(int)p->nf3, (int)p->nf2, (int)p->nf1};
}

void do_fft(FINUFFT_PLAN p, CPX *fwBatch) {
#ifdef FINUFFT_USE_DUCC0
  size_t nthreads = min<size_t>(MY_OMP_GET_MAX_THREADS(), p->opts.nthreads);
  auto ns         = gridsize_for_fft(p);
  vector<size_t> arrdims, axes;
  arrdims.push_back(size_t(p->batchSize));
  arrdims.push_back(size_t(ns[0]));
  axes.push_back(1);
  if (p->dim >= 2) {
    arrdims.push_back(size_t(ns[1]));
    axes.push_back(2);
  }
  if (p->dim >= 3) {
    arrdims.push_back(size_t(ns[2]));
    axes.push_back(3);
  }
  ducc0::vfmav<CPX> data(fwBatch, arrdims);
  if (p->dim == 1)        // 1D: no chance for FFT shortcuts
    ducc0::c2c(data, data, axes, p->fftSign < 0, FLT(1), nthreads);
  else if (p->dim == 2) { // 2D: do partial FFTs
    if (p->ms < 2)        // something is weird, do standard FFT
      ducc0::c2c(data, data, axes, p->fftSign < 0, FLT(1), nthreads);
    else {
      size_t y_lo = size_t((p->ms + 1) / 2);
      size_t y_hi = size_t(ns[1] - p->ms / 2);
      auto sub1   = ducc0::subarray(data, {{}, {}, {0, y_lo}});
      auto sub2   = ducc0::subarray(data, {{}, {}, {y_hi, ducc0::MAXIDX}});
      if (p->type == 1) // spreading, not all parts of the output array are needed
        // do axis 2 in full
        ducc0::c2c(data, data, {2}, p->fftSign < 0, FLT(1), nthreads);
      // do only parts of axis 1
      ducc0::c2c(sub1, sub1, {1}, p->fftSign < 0, FLT(1), nthreads);
      ducc0::c2c(sub2, sub2, {1}, p->fftSign < 0, FLT(1), nthreads);
      if (p->type == 2) // interpolation, parts of the input array are zero
        // do axis 2 in full
        ducc0::c2c(data, data, {2}, p->fftSign < 0, FLT(1), nthreads);
    }
  } else {                          // 3D
    if ((p->ms < 2) || (p->mt < 2)) // something is weird, do standard FFT
      ducc0::c2c(data, data, axes, p->fftSign < 0, FLT(1), nthreads);
    else {
      size_t z_lo = size_t((p->ms + 1) / 2);
      size_t z_hi = size_t(ns[2] - p->ms / 2);
      size_t y_lo = size_t((p->mt + 1) / 2);
      size_t y_hi = size_t(ns[1] - p->mt / 2);
      auto sub1   = ducc0::subarray(data, {{}, {}, {}, {0, z_lo}});
      auto sub2   = ducc0::subarray(data, {{}, {}, {}, {z_hi, ducc0::MAXIDX}});
      auto sub3   = ducc0::subarray(sub1, {{}, {}, {0, y_lo}, {}});
      auto sub4   = ducc0::subarray(sub1, {{}, {}, {y_hi, ducc0::MAXIDX}, {}});
      auto sub5   = ducc0::subarray(sub2, {{}, {}, {0, y_lo}, {}});
      auto sub6   = ducc0::subarray(sub2, {{}, {}, {y_hi, ducc0::MAXIDX}, {}});
      if (p->type == 1) { // spreading, not all parts of the output array are needed
        // do axis 3 in full
        ducc0::c2c(data, data, {3}, p->fftSign < 0, FLT(1), nthreads);
        // do only parts of axis 2
        ducc0::c2c(sub1, sub1, {2}, p->fftSign < 0, FLT(1), nthreads);
        ducc0::c2c(sub2, sub2, {2}, p->fftSign < 0, FLT(1), nthreads);
      }
      // do even smaller parts of axis 1
      ducc0::c2c(sub3, sub3, {1}, p->fftSign < 0, FLT(1), nthreads);
      ducc0::c2c(sub4, sub4, {1}, p->fftSign < 0, FLT(1), nthreads);
      ducc0::c2c(sub5, sub5, {1}, p->fftSign < 0, FLT(1), nthreads);
      ducc0::c2c(sub6, sub6, {1}, p->fftSign < 0, FLT(1), nthreads);
      if (p->type == 2) { // interpolation, parts of the input array are zero
        // do only parts of axis 2
        ducc0::c2c(sub1, sub1, {2}, p->fftSign < 0, FLT(1), nthreads);
        ducc0::c2c(sub2, sub2, {2}, p->fftSign < 0, FLT(1), nthreads);
        // do axis 3 in full
        ducc0::c2c(data, data, {3}, p->fftSign < 0, FLT(1), nthreads);
      }
    }
  }
#else
  p->fftwPlan.execute(); // if thisBatchSize<batchSize it wastes some flops
#endif
}
