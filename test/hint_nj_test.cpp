// Cases covered:
// A) If opts.upsampfac != 0 (1.25 or 2.00), hint_nj is ignored (all types).
// B) If opts.upsampfac == 0 and hint_nj == 0 (types 1 & 2), upsampfac is chosen at the
//    first setpts from the observed density and may change on any following setpts.
// C) If opts.upsampfac == 0 and hint_nj > 0 (types 1 & 2), upsampfac is chosen at
//    makeplan from density = hint_nj / N, and may change on any following setpts based
//    on the actual nj (i.e., heuristic re-evaluation).
// D) Type 3 ignores hint_nj entirely; upsampfac is chosen inside setpts from geometry-
//    based density and may change on any following setpts.
//
//    and may be recomputed if nj changes.
//

#include <finufft/heuristics.hpp>
#include <finufft/test_defs.h>

#include <cmath>
#include <cstdio>
#include <vector>
#include <type_traits>
#include <limits>

using finufft::heuristics::bestUpsamplingFactor;

static int g_debug = 1;

// -------- precision-dependent globals --------
// Tolerance used for numerical equality checks (assert), depends on FLT.
static const double g_eps = (sizeof(FLT) == 4 ? 1e-6 : 1e-12);

// Choose NUFFT accuracy tol per precision.
// float: pick > 1e-8 (and also > eps_float*100 ~ 1e-5) to avoid the clamp-to-2.0 path.
// double: pick 1e-8 (> 1e-9) to avoid the clamp-to-2.0 path.
static inline FLT get_tol() {
  if constexpr (std::is_same<FLT, float>::value) {
    return (FLT)2e-5;  // > 1e-8 and > ~1.2e-5
  } else {
    return (FLT)1e-8;  // > 1e-9
  }
}

// --------------------------------------------

static void fill_rand(FLT *a, BIGINT n, FLT scale = (FLT)PI) {
  for (BIGINT i = 0; i < n; ++i) a[i] = scale * randm11();
}

static int fail(const char *where, const char *msg) {
  std::printf("[FAIL] %s: %s\n", where, msg);
  return 1;
}

static int assert_eq_double(const char *where, const char *what, double got,
                            double expect, double eps = g_eps) {
  if (std::fabs(got - expect) > eps) {
    std::printf("[FAIL] %s: %s %.12g != %.12g (|diff|=%.3g > %.3g)\n",
                where, what, got, expect, std::fabs(got - expect), eps);
    return 1;
  }
  if (g_debug) {
    std::printf("[ OK ] %s: %s = %.6f\n", where, what, got);
  }
  return 0;
}

// ---------- Type 1 & 2 (2D) ----------
// We choose N = 8x8 => N() = 64 so that crossing density thresholds is easy:
//   density = nj / N() -> with nj = 64   => density=1  (often 1.25)
//                           nj = 4096 => density=64 (often 2.0)
static int test_type12_forced(int type, double forced_usf) {
  const char *WHERE = (type == 1 ? "type1_forced_2D" : "type2_forced_2D");
  const int dim     = 2;
  const FLT tol     = get_tol();

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.debug     = g_debug - 1;
  opts.nthreads  = 1;    // single-thread tables
  opts.hint_nj   = 4000; // ignored when upsampfac is forced
  opts.upsampfac = forced_usf;

  BIGINT Nm[2] = {8, 8}; // 2D grid: N()=64
  FINUFFT_PLAN plan;
  int ier = FINUFFT_MAKEPLAN(type, dim, Nm, +1, 1, tol, &plan, &opts);
  if (ier) return fail(WHERE, "makeplan error");

  if (int e = assert_eq_double(WHERE, "plan upsampfac", plan->opts.upsampfac, forced_usf))
  { FINUFFT_DESTROY(plan); return e; }

  // setpts #1 (small density) - remains forced
  BIGINT nj1 = 64; // density=1
  std::vector<FLT> x1(nj1), y1(nj1);
  fill_rand(x1.data(), nj1);
  fill_rand(y1.data(), nj1);
  ier = FINUFFT_SETPTS(plan, nj1, x1.data(), y1.data(), nullptr,
                       0, nullptr, nullptr, nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #1 error"); }
  if (int e = assert_eq_double(WHERE, "setpts #1 upsampfac", plan->opts.upsampfac,
                               forced_usf))
  { FINUFFT_DESTROY(plan); return e; }

  // setpts #2 (large density) - still forced
  BIGINT nj2 = 4096; // density=64
  std::vector<FLT> x2(nj2), y2(nj2);
  fill_rand(x2.data(), nj2);
  fill_rand(y2.data(), nj2);
  ier = FINUFFT_SETPTS(plan, nj2, x2.data(), y2.data(), nullptr,
                       0, nullptr, nullptr, nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #2 error"); }
  if (int e = assert_eq_double(WHERE, "setpts #2 upsampfac", plan->opts.upsampfac,
                               forced_usf))
  { FINUFFT_DESTROY(plan); return e; }

  FINUFFT_DESTROY(plan);
  return 0;
}

static int test_type12_hint0(int type) {
  const char *WHERE = (type == 1 ? "type1_hint0_2D" : "type2_hint0_2D");
  const int dim     = 2;
  const FLT tol     = get_tol();

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.debug     = g_debug - 1;
  opts.nthreads  = 1;
  opts.hint_nj   = 0;     // no hint
  opts.upsampfac = 0.0;   // heuristic decides at first setpts

  BIGINT Nm[2] = {8, 8};  // N() = 64
  FINUFFT_PLAN plan;
  int ier = FINUFFT_MAKEPLAN(type, dim, Nm, +1, 1, tol, &plan, &opts);
  if (ier) return fail(WHERE, "makeplan error");

  if (int e = assert_eq_double(WHERE, "plan-time upsampfac", plan->opts.upsampfac, 0.0))
  { FINUFFT_DESTROY(plan); return e; }

  // First setpts: smaller density
  BIGINT nj1 = 64; // density=1
  std::vector<FLT> x1(nj1), y1(nj1);
  fill_rand(x1.data(), nj1);
  fill_rand(y1.data(), nj1);
  ier = FINUFFT_SETPTS(plan, nj1, x1.data(), y1.data(), nullptr,
                       0, nullptr, nullptr, nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #1 error"); }

  double density1 = double(nj1) / double(plan->N());
  double expect1 =
      bestUpsamplingFactor<FLT>(opts.nthreads, (FLT)density1, dim, type, tol);
  if (int e = assert_eq_double(WHERE, "post-setpts #1 upsampfac",
                               plan->opts.upsampfac, expect1))
  { FINUFFT_DESTROY(plan); return e; }

  // Second setpts: much larger density
  BIGINT nj2 = 4096; // density=64
  std::vector<FLT> x2(nj2), y2(nj2);
  fill_rand(x2.data(), nj2);
  fill_rand(y2.data(), nj2);
  ier = FINUFFT_SETPTS(plan, nj2, x2.data(), y2.data(), nullptr,
                       0, nullptr, nullptr, nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #2 error"); }

  double density2 = double(nj2) / double(plan->N());
  double expect2 =
      bestUpsamplingFactor<FLT>(opts.nthreads, (FLT)density2, dim, type, tol);
  if (int e = assert_eq_double(WHERE, "post-setpts #2 upsampfac (may change)",
                               plan->opts.upsampfac, expect2))
  { FINUFFT_DESTROY(plan); return e; }

  FINUFFT_DESTROY(plan);
  return 0;
}

static int test_type12_hintpos(int type) {
  const char *WHERE = (type == 1 ? "type1_hintpos_2D" : "type2_hintpos_2D");
  const int dim     = 2;
  const FLT tol     = get_tol();

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.debug     = g_debug - 1;
  opts.nthreads  = 1;
  opts.hint_nj   = 128;   // density_hint = 128/64 = 2 (often 1.25)
  opts.upsampfac = 0.0;   // decide using hint at makeplan

  BIGINT Nm[2] = {8, 8};  // N()=64
  FINUFFT_PLAN plan;
  int ier = FINUFFT_MAKEPLAN(type, dim, Nm, +1, 1, tol, &plan, &opts);
  if (ier) return fail(WHERE, "makeplan error");

  double density_hint = double(opts.hint_nj) / double(plan->N());
  double expect_plan =
      bestUpsamplingFactor<FLT>(opts.nthreads, (FLT)density_hint, dim, type, tol);
  if (int e = assert_eq_double(WHERE, "plan-time upsampfac from hint",
                               plan->opts.upsampfac, expect_plan))
  { FINUFFT_DESTROY(plan); return e; }

  // First setpts with nj == hint_nj -> shouldn't change
  BIGINT nj1 = opts.hint_nj;
  std::vector<FLT> x1(nj1), y1(nj1);
  fill_rand(x1.data(), nj1);
  fill_rand(y1.data(), nj1);
  ier = FINUFFT_SETPTS(plan, nj1, x1.data(), y1.data(), nullptr,
                       0, nullptr, nullptr, nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #1 error"); }
  if (int e = assert_eq_double(WHERE, "post-setpts #1 upsampfac (nj==hint)",
                               plan->opts.upsampfac, expect_plan))
  { FINUFFT_DESTROY(plan); return e; }

  // Second setpts with large nj to cross thresholds (may update)
  BIGINT nj2 = 4096; // density=64
  std::vector<FLT> x2(nj2), y2(nj2);
  fill_rand(x2.data(), nj2);
  fill_rand(y2.data(), nj2);
  ier = FINUFFT_SETPTS(plan, nj2, x2.data(), y2.data(), nullptr,
                       0, nullptr, nullptr, nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #2 error"); }

  double density2 = double(nj2) / double(plan->N());
  double expect2 =
      bestUpsamplingFactor<FLT>(opts.nthreads, (FLT)density2, dim, type, tol);
  if (int e = assert_eq_double(WHERE, "post-setpts #2 upsampfac (nj!=hint)",
                               plan->opts.upsampfac, expect2))
  { FINUFFT_DESTROY(plan); return e; }

  FINUFFT_DESTROY(plan);
  return 0;
}

// ---------- Type 3 (2D) ----------
static int test_type3_forced(double forced_usf) {
  const char *WHERE = "type3_forced_2D";
  const int dim     = 2;
  const FLT tol     = get_tol();

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.debug     = g_debug - 1;
  opts.nthreads  = 1;
  opts.hint_nj   = 4000;       // ignored in type 3 when upsampfac!=0
  opts.upsampfac = forced_usf; // forced

  BIGINT Nm[2] = {0, 0};       // type-3 makeplan ignores Nm
  FINUFFT_PLAN plan;
  int ier = FINUFFT_MAKEPLAN(3, dim, Nm, +1, 1, tol, &plan, &opts);
  if (ier) return fail(WHERE, "makeplan error");

  if (int e = assert_eq_double(WHERE, "plan-time upsampfac", plan->opts.upsampfac,
                               forced_usf))
  { FINUFFT_DESTROY(plan); return e; }

  BIGINT nk = 16; // some freqs
  std::vector<FLT> s(nk), t(nk);
  fill_rand(s.data(), nk);
  fill_rand(t.data(), nk);

  // setpts #1: small nj, moderate bandwidth
  BIGINT nj1 = 64;
  std::vector<FLT> x1(nj1), y1(nj1);
  fill_rand(x1.data(), nj1, (FLT)(PI * 8));  // moderate width
  fill_rand(y1.data(), nj1, (FLT)(PI * 8));
  ier = FINUFFT_SETPTS(plan, nj1, x1.data(), y1.data(), nullptr,
                       nk, s.data(), t.data(), nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #1 error"); }
  if (int e = assert_eq_double(WHERE, "post-setpts #1 upsampfac (forced)",
                               plan->opts.upsampfac, forced_usf))
  { FINUFFT_DESTROY(plan); return e; }

  // setpts #2: very different nj/bandwidth, still forced
  BIGINT nj2 = 4096;
  std::vector<FLT> x2(nj2), y2(nj2);
  fill_rand(x2.data(), nj2, (FLT)(PI * 0.1)); // very small width
  fill_rand(y2.data(), nj2, (FLT)(PI * 0.1));
  ier = FINUFFT_SETPTS(plan, nj2, x2.data(), y2.data(), nullptr,
                       nk, s.data(), t.data(), nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #2 error"); }
  if (int e = assert_eq_double(WHERE, "post-setpts #2 upsampfac (forced)",
                               plan->opts.upsampfac, forced_usf))
  { FINUFFT_DESTROY(plan); return e; }

  FINUFFT_DESTROY(plan);
  return 0;
}

// For type 3 with heuristic (upsampfac==0): hint ignored; upsampfac selected in setpts.
// NOTE: bestUpsamplingFactor currently returns 1.25 for type-3, independent of density.
// We still vary geometry (bandwidth) between calls to prepare for future changes.
static int test_type3_hint_ignored() {
  const char *WHERE = "type3_hint_ignored_2D";
  const int dim     = 2;
  const FLT tol     = get_tol();

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.debug     = g_debug - 1;
  opts.nthreads  = 1;
  opts.hint_nj   = 4000; // ignored
  opts.upsampfac = 0.0;  // heuristic decides in setpts

  BIGINT Nm[2] = {0, 0};
  FINUFFT_PLAN plan;
  int ier = FINUFFT_MAKEPLAN(3, dim, Nm, +1, 1, tol, &plan, &opts);
  if (ier) return fail(WHERE, "makeplan error");

  if (int e = assert_eq_double(WHERE, "plan-time upsampfac", plan->opts.upsampfac, 0.0))
  { FINUFFT_DESTROY(plan); return e; }

  BIGINT nk = 16;
  std::vector<FLT> s(nk), t(nk);
  fill_rand(s.data(), nk);
  fill_rand(t.data(), nk);

  // setpts #1: large bandwidth
  BIGINT nj1 = 256;
  std::vector<FLT> x1(nj1), y1(nj1);
  fill_rand(x1.data(), nj1, (FLT)(PI * 64));  // very wide
  fill_rand(y1.data(), nj1, (FLT)(PI * 64));
  ier = FINUFFT_SETPTS(plan, nj1, x1.data(), y1.data(), nullptr,
                       nk, s.data(), t.data(), nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #1 error"); }

  double density1 = double(nj1) / double(plan->N());
  double expect1  = bestUpsamplingFactor<FLT>(opts.nthreads, (FLT)density1, dim, 3, tol);
  if (int e = assert_eq_double(WHERE, "post-setpts #1 upsampfac",
                               plan->opts.upsampfac, expect1))
  { FINUFFT_DESTROY(plan); return e; }

  // setpts #2: tiny bandwidth (should not change under current heuristic)
  BIGINT nj2 = 512;
  std::vector<FLT> x2(nj2), y2(nj2);
  fill_rand(x2.data(), nj2, (FLT)(PI * 0.01)); // very narrow
  fill_rand(y2.data(), nj2, (FLT)(PI * 0.01));
  ier = FINUFFT_SETPTS(plan, nj2, x2.data(), y2.data(), nullptr,
                       nk, s.data(), t.data(), nullptr);
  if (ier) { FINUFFT_DESTROY(plan); return fail(WHERE, "setpts #2 error"); }

  double density2 = double(nj2) / double(plan->N());
  double expect2  = bestUpsamplingFactor<FLT>(opts.nthreads, (FLT)density2, dim, 3, tol);
  if (int e = assert_eq_double(WHERE, "post-setpts #2 upsampfac (may change in future)",
                               plan->opts.upsampfac, expect2))
  { FINUFFT_DESTROY(plan); return e; }

  FINUFFT_DESTROY(plan);
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1) g_debug = std::atoi(argv[1]);

  // Type 1 & 2: forced upsampfac (hint ignored)
  if (int e = test_type12_forced(1, 2.00)) return e;
  if (int e = test_type12_forced(2, 2.00)) return e;

  // Type 1 & 2 (2D): hint==0 -> decide at first setpts, may change on next
  if (int e = test_type12_hint0(1)) return e;
  if (int e = test_type12_hint0(2)) return e;

  // Type 1 & 2 (2D): hint>0 -> decide at makeplan, may update if nj != hint
  if (int e = test_type12_hintpos(1)) return e;
  if (int e = test_type12_hintpos(2)) return e;

  // Type 3 (2D): forced upsampfac (hint ignored)
  if (int e = test_type3_forced(2.00)) return e;

  // Type 3 (2D): hint ignored, heuristic decides inside setpts; we vary bandwidth
  if (int e = test_type3_hint_ignored()) return e;

  if (g_debug) std::printf("All tests passed.\n");
  return 0;
}
