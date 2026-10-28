/* Small C test to exercise deterministic error paths in the C API.
   Prints concise PASS on success and non-zero exit on mismatch.
*/
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <finufft.h>
#include <finufft_errors.h>

static void dummy_lock(void *data) { (void)data; }

int main() {
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.allow_eps_too_small = 1;

  finufft_plan plan = NULL;
  int rc            = 0;

  // modes array for makeplan (only first 'dim' entry used)
  int64_t N[3] = {10, 10, 10};

  // Invalid type -> expect FINUFFT_ERR_TYPE_NOTVALID
  rc = finufft_makeplan(4, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_TYPE_NOTVALID) {
    fprintf(stderr, "type invalid: expected %d got %d\n", FINUFFT_ERR_TYPE_NOTVALID, rc);
    return 2;
  }

  // Invalid dim -> expect FINUFFT_ERR_DIM_NOTVALID
  rc = finufft_makeplan(1, 4, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_DIM_NOTVALID) {
    fprintf(stderr, "dim invalid: expected %d got %d\n", FINUFFT_ERR_DIM_NOTVALID, rc);
    return 3;
  }

  // Invalid ntrans (zero) -> expect FINUFFT_ERR_NTRANS_NOTVALID
  rc = finufft_makeplan(1, 1, N, 1, 0, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_NTRANS_NOTVALID) {
    fprintf(stderr, "ntrans invalid: expected %d got %d\n", FINUFFT_ERR_NTRANS_NOTVALID,
            rc);
    return 4;
  }

  // Make a valid plan, then exercise setpts with negative nj
  rc = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc) {
    fprintf(stderr, "makeplan (valid) failed: rc=%d\n", rc);
    return 5;
  }

  int64_t neg_nj = -1;
  rc             = finufft_setpts(plan, neg_nj, NULL, NULL, NULL, 0, NULL, NULL, NULL);
  if (rc != FINUFFT_ERR_NUM_NU_PTS_INVALID) {
    fprintf(stderr, "setpts negative nj: expected %d got %d\n",
            FINUFFT_ERR_NUM_NU_PTS_INVALID, rc);
    finufft_destroy(plan);
    return 6;
  }

  finufft_destroy(plan);

#ifdef _OPENMP
  // Invalid lock function pairing -> expect FINUFFT_ERR_LOCK_FUNS_INVALID
  finufft_default_opts(&opts);
  opts.allow_eps_too_small = 1;
  opts.fftw_lock_fun   = dummy_lock;
  opts.fftw_unlock_fun = NULL;
  rc                   = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_LOCK_FUNS_INVALID) {
    fprintf(stderr, "lock funs invalid: expected %d got %d\n",
            FINUFFT_ERR_LOCK_FUNS_INVALID, rc);
    return 7;
  }
#endif

  // Invalid upsampfac -> expect FINUFFT_ERR_UPSAMPFAC_TOO_SMALL
  finufft_default_opts(&opts);
  opts.allow_eps_too_small = 1;
  opts.upsampfac = 1.0;
  rc             = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_UPSAMPFAC_TOO_SMALL) {
    fprintf(stderr, "upsampfac too small: expected %d got %d\n",
            FINUFFT_ERR_UPSAMPFAC_TOO_SMALL, rc);
    return 9;
  }

  // Invalid kernel formula -> expect FINUFFT_ERR_KERFORMULA_NOTVALID.
  // Only the PSWF formulas (7..10) survive; 1..6 (legacy ES/KB/cosh) and any other
  // out-of-range value are now rejected.
  // 1 is the legacy ES value documented up to v2.5.0, so users will try it.
  const int bad_kerformulas[] = {1, 3, 6, 11, 99};
  for (size_t i = 0; i < sizeof(bad_kerformulas) / sizeof(bad_kerformulas[0]); ++i) {
    finufft_default_opts(&opts);
    opts.allow_eps_too_small = 1;
    opts.upsampfac = 2.0; // force kernel setup in makeplan
    opts.spread_kerformula = bad_kerformulas[i];
    rc = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
    if (rc != FINUFFT_ERR_KERFORMULA_NOTVALID) {
      fprintf(stderr, "kerformula=%d invalid: expected %d got %d\n", bad_kerformulas[i],
              FINUFFT_ERR_KERFORMULA_NOTVALID, rc);
      return 10;
    }
  }

  // Kernel formula 10 (beta from nf/N) is valid -> expect success
  finufft_default_opts(&opts);
  opts.allow_eps_too_small = 1;
  opts.upsampfac           = 2.0; // force kernel setup in makeplan
  opts.spread_kerformula   = 10;
  rc                       = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != 0) {
    fprintf(stderr, "kerformula=10: expected 0 got %d\n", rc);
    return 12;
  }
  finufft_destroy(plan);

  // Kernel formula 10 recomputes beta from nf/N. N = 103 at upsampfac 1.25 rounds
  // up to nf = 144, so kf 10 must give different output than kf 8 there.
  {
    enum { NJ = 100, NM = 103 };
    double x[NJ];
    double _Complex c[NJ], f8[NM], f10[NM];
    for (int j = 0; j < NJ; ++j) {
      x[j] = -3.0 + 6.0 * j / NJ;
      c[j] = cos(0.7 * j) + I * sin(1.3 * j);
    }
    int rc8, rc10;
    finufft_default_opts(&opts);
    opts.allow_eps_too_small = 1;
    opts.upsampfac           = 1.25;
    opts.nthreads            = 1; // threaded spreading sums in varying order
    opts.spread_kerformula   = 8;
    rc8                      = finufft1d1(NJ, x, c, 1, 1e-6, NM, f8, &opts);
    opts.spread_kerformula   = 10;
    rc10                     = finufft1d1(NJ, x, c, 1, 1e-6, NM, f10, &opts);
    if (rc8 != 0 || rc10 != 0) {
      fprintf(stderr, "kerformula 8/10 transforms: expected 0 got %d/%d\n", rc8, rc10);
      return 13;
    }
    if (memcmp(f8, f10, sizeof(f8)) == 0) {
      fprintf(stderr, "kerformula=10: output equals kerformula=8 at nf/N != upsampfac\n");
      return 14;
    }
  }

  // Deprecated opts are ignored, not errors, but must warn at runtime (the only
  // signal non-C++ wrappers get). ctest greps stderr for these three lines.
  finufft_default_opts(&opts);
  opts.allow_eps_too_small = 1;
  opts.spread_thread = 3;
  opts.spread_kerevalmeth = 0;
  opts.spread_kerpad = 0;
  rc = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != 0) {
    fprintf(stderr, "deprecated opts: expected 0 got %d\n", rc);
    return 11;
  }
  finufft_destroy(plan);

  printf("error_handling: PASS\n");
  return 0;
}
