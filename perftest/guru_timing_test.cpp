#include <finufft/test_defs.h>
// for sleep call
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include <Windows.h>
void sleep(unsigned long seconds) { Sleep(seconds * 1000); }
#else
#include <unistd.h>
#endif
using namespace std;
using namespace finufft;
using namespace finufft::utils;

// forward declaration of helper to (repeatedly if needed) call finufft?d?
double many_simple_calls(CPX *c, CPX *F, FLT *x, FLT *y, FLT *z, FINUFFT_PLAN plan);

// --------------------------------------------------------------------------
int main(int argc, char *argv[])
/* Timing-only tester for the guru interface, allowing control of many params
   and opts from the command line.
   It compares doing many transforms with same NU pts, with repeated calls to
   the simple interface.

   This is pretty old clunky code, not a main part of self-test, and
   need not be maintained.

   Warning: unlike the finufft?d{many}_test routines, this does *not* perform
   a math test of the library, just consistency of the simple vs guru
   interfaces, and measuring their speed ratio.

   Usage: guru_timing_test ntransf type ndim Nmodes1 Nmodes2 Nmodes3 Nsrc
          [tol [debug [spread_thread [maxbatchsize [spread_sort [upsampfac]]]]]]

   debug = 0: rel errors and overall timing
           1: timing breakdowns
           2: also spreading output

   spread_scheme = 0: sequential maximally multithreaded spread/interp
                   1: parallel singlethreaded spread/interp, nested last batch

   Example: guru_timing_test 100 1 2 100 100 0 1000000 1e-3 1 0 0 2 2.0

   The unused dimensions of Nmodes may be left as zero.
   For type 3, Nmodes{1,2,3} controls the spread of NU freq targs in each dim.
   Example w/ nk = 5000: finufftGuru_test 1 3 2 100 50 0 1000000 1e-12 0

   By: Andrea Malleo 2019. Tidied and simplified by Alex Barnett 2020.
   added 2 extra args, 5/22/20. Moved to perftests 7/23/20.
*/
{
  double tsleep = 0.1;         // how long wait between tests to let FFTW settle (1.0?)
  int ntransf, type, ndim;
  BIGINT M, N1, N2, N3;        // M = # srcs, N1,N2,N3= # modes in each dim
  double w, tol = 1e-6;
  int isign = +1;              // choose which exponential sign to test
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts); // for guru interface

  // Collect command line arguments ------------------------------------------
  if (argc < 8 || argc > 14) {
    fprintf(
        stderr,
        "Usage: guru_timing_test ntransf type ndim N1 N2 N3 Nsrc [tol [debug "
        "[spread_thread [maxbatchsize [spread_sort "
        "[upsampfac]]]]]]\n\teg:\tguru_timing_test 100 1 2 1e2 1e2 0 1e6 1e-3 1 0 0 2\n");
    return 1;
  }
  sscanf(argv[1], "%d", &ntransf);
  sscanf(argv[2], "%d", &type);
  sscanf(argv[3], "%d", &ndim);
  sscanf(argv[4], "%lf", &w);
  N1 = (BIGINT)w;
  sscanf(argv[5], "%lf", &w);
  N2 = (BIGINT)w;
  sscanf(argv[6], "%lf", &w);
  N3 = (BIGINT)w;
  sscanf(argv[7], "%lf", &w);
  M = (BIGINT)w;
  if (argc > 8) sscanf(argv[8], "%lf", &tol);
  if (argc > 9) sscanf(argv[9], "%d", &opts.debug);
  opts.spread_debug = (opts.debug > 1) ? 1 : 0; // see output from spreader
  if (argc > 10) sscanf(argv[10], "%d", &opts.spread_thread);
  if (argc > 11) sscanf(argv[11], "%d", &opts.maxbatchsize);
  if (argc > 12) sscanf(argv[12], "%d", &opts.spread_sort);
  if (argc > 13) {
    sscanf(argv[13], "%lf", &w);
    opts.upsampfac = (FLT)w;
  }

  // Allocate and initialize input -------------------------------------------
  cout << scientific << setprecision(15);
  N2       = (N2 == 0) ? 1 : N2;
  N3       = (N3 == 0) ? 1 : N3;
  BIGINT N = N1 * N2 * N3;

  FLT *s = NULL;
  FLT *t = NULL;
  FLT *u = NULL;
  if (type == 3) { // make target freq NU pts for type 3 (N of them)...
    s      = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (1-cmpt)
    FLT S1 = (FLT)N1 / 2;
#pragma omp parallel
    {
      unsigned int se = MY_OMP_GET_THREAD_NUM(); // needed for parallel random #s
#pragma omp for schedule(dynamic, TEST_RANDCHUNK)
      for (BIGINT k = 0; k < N; ++k) {
        s[k] = S1 * (1.7 + randm11r(&se)); // note the offset, to test type 3.
      }
      if (ndim > 1) {
        t      = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (2-cmpt)
        FLT S2 = (FLT)N2 / 2;
#pragma omp for schedule(dynamic, TEST_RANDCHUNK)
        for (BIGINT k = 0; k < N; ++k) {
          t[k] = S2 * (-0.5 + randm11r(&se));
        }
      }
      if (ndim > 2) {
        u      = (FLT *)malloc(sizeof(FLT) * N); // targ freqs (3-cmpt)
        FLT S3 = (FLT)N3 / 2;
#pragma omp for schedule(dynamic, TEST_RANDCHUNK)
        for (BIGINT k = 0; k < N; ++k) {
          u[k] = S3 * (0.9 + randm11r(&se));
        }
      }
    }
  }

  CPX *c = (CPX *)malloc(sizeof(CPX) * M * ntransf);             // strengths
  CPX *F = (CPX *)malloc(sizeof(CPX) * N * ntransf);             // mode ampls

  FLT *x = (FLT *)malloc(sizeof(FLT) * M), *y = NULL, *z = NULL; // NU pts x coords
  if (ndim > 1) y = (FLT *)malloc(sizeof(FLT) * M);              // NU pts y coords
  if (ndim > 2) z = (FLT *)malloc(sizeof(FLT) * M);              // NU pts z coords
#pragma omp parallel
  {
    unsigned int se = MY_OMP_GET_THREAD_NUM(); // needed for parallel random #s
#pragma omp for schedule(dynamic, TEST_RANDCHUNK)
    for (BIGINT j = 0; j < M; ++j) {
      x[j] = M_PI * randm11r(&se);
      if (y) y[j] = M_PI * randm11r(&se);
      if (z) z[j] = M_PI * randm11r(&se);
    }
#pragma omp for schedule(dynamic, TEST_RANDCHUNK)
    for (BIGINT i = 0; i < ntransf * M; i++) // random strengths
      c[i] = crandm11r(&se);
  }

  // Andrea found the following are needed to get reliable independent timings:
  FFTW_CLEANUP();
  FFTW_CLEANUP_THREADS();
  FFTW_FORGET_WISDOM();
  // std::this_thread::sleep_for(std::chrono::seconds(1));
  sleep(tsleep);

  printf("FINUFFT %dd%d use guru interface to do %d calls together:-------------------\n",
         ndim, type, ntransf);
  FINUFFT_PLAN plan;                // instantiate a finufft_plan
  finufft::utils::CNTime timer;
  timer.start();                    // Guru Step 1
  BIGINT n_modes[3] = {N1, N2, N3}; // #modes per dimension (ignored for t3)
  int ier = FINUFFT_MAKEPLAN(type, ndim, n_modes, isign, ntransf, tol, &plan, &opts);
  // (NB: the opts struct can no longer be modified with effect!)
  double plan_t = timer.elapsedsec();
  if (ier > 1) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else {
    if (type != 3)
      printf("\tplan, for %lld modes: \t\t%.3g s\n", (long long)N, plan_t);
    else
      printf("\tplan:\t\t\t\t\t%.3g s\n", plan_t);
  }

  timer.restart();                                              // Guru Step 2
  ier           = FINUFFT_SETPTS(plan, M, x, y, z, N, s, t, u); //(t1,2: N,s,t,u ignored)
  double sort_t = timer.elapsedsec();
  if (ier) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else {
    if (type != 3)
      printf("\tsetpts for %lld NU pts: \t\t%.3g s\n", (long long)M, sort_t);
    else
      printf("\tsetpts for %lld + %lld NU pts: \t%.3g s\n", (long long)M, (long long)N,
             sort_t);
  }

  timer.restart(); // Guru Step 3
  ier           = FINUFFT_EXECUTE(plan, c, F);
  double exec_t = timer.elapsedsec();
  if (ier) {
    printf("error (ier=%d)!\n", ier);
    return ier;
  } else
    printf("\texec \t\t\t\t\t%.3g s\n", exec_t);

  double totalTime = plan_t + sort_t + exec_t;
  if (type != 3)
    printf("ntr=%d: %lld NU pts to %lld modes in %.3g s \t%.3g NU pts/s\n", ntransf,
           (long long)M, (long long)N, totalTime, ntransf * M / totalTime);
  else
    printf("ntr=%d: %lld NU pts to %lld NU pts in %.3g s \t%.3g tot NU pts/s\n", ntransf,
           (long long)M, (long long)N, totalTime, ntransf * (N + M) / totalTime);

  // Comparing timing results with repeated calls to corresponding finufft function...

  // The following would normally be done between independent timings, as found
  // by Andrea Malleo, but in this case we need to access the plan later
  // for many_simple_calls() to work, so we cannot do FFTW cleanup without
  // apparently causing segfault :(. So we skip them.
  // FFTW_CLEANUP();
  // FFTW_CLEANUP_THREADS();
  // FFTW_FORGET_WISDOM();

  // std::this_thread::sleep_for(std::chrono::seconds(1)); if c++11 is allowed
  sleep(tsleep); // sleep for one second using linux sleep call

  printf(
      "Compare speed of repeated calls to simple interface:------------------------\n");
  // this used to actually call Alex's old (v1.1) src/finufft?d.cpp routines.
  // Since we don't want to ship those, we now call the simple interfaces.

  double simpleTime = many_simple_calls(c, F, x, y, z, plan);
  if (isnan(simpleTime)) return 1;

  if (type != 3)
    printf("%d of:\t%lld NU pts to %lld modes in %.3g s   \t%.3g NU pts/s\n", ntransf,
           (long long)M, (long long)N, simpleTime, ntransf * M / simpleTime);
  else
    printf("%d of:\t%lld NU pts to %lld NU pts in %.3g s  \t%.3g tot NU pts/s\n", ntransf,
           (long long)M, (long long)N, simpleTime, ntransf * (M + N) / simpleTime);
  printf("\tspeedup \t T_finufft%dd%d_simple / T_finufft%dd%d = %.3g\n", ndim, type, ndim,
         type, simpleTime / totalTime);

  FINUFFT_DESTROY(plan); // Guru Step 4
  // (must be done *after* many_simple_calls, which sneaks a look at the plan!)
  // however, segfaults, maybe because plan->opts.debug changed?

  //---------------------------- Free Memory (no need to test if NULL)
  free(F);
  free(c);
  free(x);
  free(y);
  free(z);
  free(s);
  free(t);
  free(u);
  return 0;
}

// -------------------------------- HELPER FUNCS ----------------------------

double finufftFunnel(CPX *cStart, CPX *fStart, FLT *x, FLT *y, FLT *z, FINUFFT_PLAN plan)
/* Helper to make a simple interface call with parameters pulled out of a
   guru-interface plan. Reads opts from the
   finufft plan, and the pointers to various data vectors that users shouldn't
   normally access, and does a single simple interface call.
   Returns the run-time in seconds, or -1.0 if error.
   Malleo 2019; xyz passed in by Barnett 5/26/20 to prevent X_orig fields.
*/
{
  finufft::utils::CNTime timer;
  timer.start();
  int ier             = 0;
  double t            = 0;
  double fail         = NAN;           // dummy code for failure
  finufft_opts *popts = &(plan->opts); // opts ptr, as v1.2 simple calls need
  switch (plan->dim) {

  case 1: // 1D
    switch (plan->type) {

    case 1:
      timer.restart();
      ier = FINUFFT1D1(plan->nj, x, cStart, plan->fftSign, plan->tol, plan->ms, fStart,
                       popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    case 2:
      timer.restart();
      ier = FINUFFT1D2(plan->nj, x, cStart, plan->fftSign, plan->tol, plan->ms, fStart,
                       popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    case 3:
      timer.restart();
      ier = FINUFFT1D3(plan->nj, x, cStart, plan->fftSign, plan->tol, plan->nk, plan->S,
                       fStart, popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    default:
      return fail;
    }

  case 2: // 2D
    switch (plan->type) {

    case 1:
      timer.restart();
      ier = FINUFFT2D1(plan->nj, x, y, cStart, plan->fftSign, plan->tol, plan->ms,
                       plan->mt, fStart, popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    case 2:
      timer.restart();
      ier = FINUFFT2D2(plan->nj, x, y, cStart, plan->fftSign, plan->tol, plan->ms,
                       plan->mt, fStart, popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    case 3:
      timer.restart();
      ier = FINUFFT2D3(plan->nj, x, y, cStart, plan->fftSign, plan->tol, plan->nk,
                       plan->S, plan->T, fStart, popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    default:
      return fail;
    }

  case 3: // 3D
    switch (plan->type) {

    case 1:
      timer.restart();
      ier = FINUFFT3D1(plan->nj, x, y, z, cStart, plan->fftSign, plan->tol, plan->ms,
                       plan->mt, plan->mu, fStart, popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    case 2:
      timer.restart();
      ier = FINUFFT3D2(plan->nj, x, y, z, cStart, plan->fftSign, plan->tol, plan->ms,
                       plan->mt, plan->mu, fStart, popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    case 3:
      timer.restart();
      ier = FINUFFT3D3(plan->nj, x, y, z, cStart, plan->fftSign, plan->tol, plan->nk,
                       plan->S, plan->T, plan->U, fStart, popts);
      t   = timer.elapsedsec();
      if (ier)
        return fail;
      else
        return t;

    default: // invalid type
      return fail;
    }

  default: // invalid dimension
    return fail;
  }
}

double many_simple_calls(CPX *c, CPX *F, FLT *x, FLT *y, FLT *z, FINUFFT_PLAN plan)
/* A unified interface to all of the simple interfaces, with a loop over
   many such transforms. Returns total time reported by the transforms.
   (Used to call pre-v1.2 single implementations in finufft, via runOldFinufft.
   The repo no longer contains those implementations, which used to be in a
   subdirectory.)
*/
{
  CPX *cStart;
  CPX *fStart;

  double time = 0;
  double temp = 0;
  ;

  for (int k = 0; k < plan->ntrans; k++) {
    cStart = c + plan->nj * k;
    fStart = F + plan->ms * plan->mt * plan->mu * k;

    // printf("k=%d, debug=%d.................\n",k, plan->opts.debug);
    if (k != 0) { // prevent massive debug output
      plan->opts.debug        = 0;
      plan->opts.spread_debug = 0;
    }

    temp = finufftFunnel(cStart, fStart, x, y, z, plan);
    if (isnan(temp)) {
      fprintf(stderr, "[%s] Funnel call to finufft failed!\n", __func__);
      return NAN;
    } else
      time += temp;
  }
  return time;
}
