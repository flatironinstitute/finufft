/* Small C test to exercise deterministic error paths in the C API.
   Prints concise PASS on success and non-zero exit on mismatch.
*/
#include <stdio.h>
#include <stdlib.h>

#include <finufft.h>
#include <finufft_errors.h>

static void dummy_lock(void *data) { (void)data; }

int main(void) {
  finufft_opts opts;
  finufft_default_opts(&opts);

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
  opts.fftw_lock_fun   = dummy_lock;
  opts.fftw_unlock_fun = NULL;
  rc                   = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_LOCK_FUNS_INVALID) {
    fprintf(stderr, "lock funs invalid: expected %d got %d\n",
            FINUFFT_ERR_LOCK_FUNS_INVALID, rc);
    return 7;
  }
#endif

  // Invalid spread_thread -> expect FINUFFT_ERR_SPREAD_THREAD_NOTVALID
  finufft_default_opts(&opts);
  opts.spread_thread = 3;
  rc                 = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_SPREAD_THREAD_NOTVALID) {
    fprintf(stderr, "spread_thread invalid: expected %d got %d\n",
            FINUFFT_ERR_SPREAD_THREAD_NOTVALID, rc);
    return 8;
  }

  // Invalid upsampfac -> expect FINUFFT_ERR_UPSAMPFAC_TOO_SMALL
  finufft_default_opts(&opts);
  opts.upsampfac = 1.0;
  rc             = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_UPSAMPFAC_TOO_SMALL) {
    fprintf(stderr, "upsampfac too small: expected %d got %d\n",
            FINUFFT_ERR_UPSAMPFAC_TOO_SMALL, rc);
    return 9;
  }

  // Invalid kernel formula -> expect FINUFFT_ERR_KERFORMULA_NOTVALID
  finufft_default_opts(&opts);
  opts.upsampfac         = 2.0; // force kernel setup in makeplan
  opts.spread_kerformula = 99;  // invalid kernel formula
  rc                     = finufft_makeplan(1, 1, N, 1, 1, 1e-6, &plan, &opts);
  if (rc != FINUFFT_ERR_KERFORMULA_NOTVALID) {
    fprintf(stderr, "kerformula invalid: expected %d got %d\n",
            FINUFFT_ERR_KERFORMULA_NOTVALID, rc);
    return 10;
  }

  printf("error_handling: PASS\n");
  return 0;
}
