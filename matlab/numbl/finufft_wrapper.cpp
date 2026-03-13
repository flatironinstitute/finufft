// C++ wrapper around FINUFFT for WebAssembly.
// Accepts separate real/imaginary arrays for the WASM interface.
// Exports all 9 simple interface functions (1d1..3d3) and guru API.

#include <complex>
#include <cstdlib>
#include <cstring>
#include <finufft.h>

using CPX = std::complex<double>;

// ── Guru API: handle table ──────────────────────────────────────────────────

struct PlanInfo {
  finufft_plan plan;
  int type;
  // Owned copies of coordinate arrays (finufft_setpts stores pointers, not copies)
  double *xj, *yj, *zj, *s, *t, *u;
};

static PlanInfo plan_table[64];
static int plan_table_used[64] = {};
static int plan_count = 0;

static finufft_opts default_opts() {
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.nthreads = 1;
  opts.debug = 0;
  return opts;
}

static void interleave(int n, const double *re, const double *im, CPX *out) {
  for (int j = 0; j < n; j++)
    out[j] = CPX(re[j], im[j]);
}

static void deinterleave(int n, const CPX *in, double *re, double *im) {
  for (int j = 0; j < n; j++) {
    re[j] = in[j].real();
    im[j] = in[j].imag();
  }
}

extern "C" {

__attribute__((export_name("my_malloc")))
void *my_malloc(int size) { return std::malloc(size); }

__attribute__((export_name("my_free")))
void my_free(void *ptr) { std::free(ptr); }

// ── Type 1: nonuniform → uniform ────────────────────────────────────────────

__attribute__((export_name("nufft1d1")))
int nufft1d1_w(int nj, double *x,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  auto *fk = new CPX[ms];
  auto opts = default_opts();
  int ier = finufft1d1((int64_t)nj, x, cj, iflag, eps, (int64_t)ms, fk, &opts);
  deinterleave(ms, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft2d1")))
int nufft2d1_w(int nj, double *x, double *y,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms, int mt,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  int nout = ms * mt;
  auto *fk = new CPX[nout];
  auto opts = default_opts();
  int ier = finufft2d1((int64_t)nj, x, y, cj, iflag, eps,
                       (int64_t)ms, (int64_t)mt, fk, &opts);
  deinterleave(nout, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft3d1")))
int nufft3d1_w(int nj, double *x, double *y, double *z,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms, int mt, int mu,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  int nout = ms * mt * mu;
  auto *fk = new CPX[nout];
  auto opts = default_opts();
  int ier = finufft3d1((int64_t)nj, x, y, z, cj, iflag, eps,
                       (int64_t)ms, (int64_t)mt, (int64_t)mu, fk, &opts);
  deinterleave(nout, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

// ── Type 2: uniform → nonuniform ────────────────────────────────────────────

__attribute__((export_name("nufft1d2")))
int nufft1d2_w(int nj, double *x,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms,
               double *fk_re, double *fk_im) {
  auto *fk = new CPX[ms];
  interleave(ms, fk_re, fk_im, fk);
  auto *cj = new CPX[nj];
  auto opts = default_opts();
  int ier = finufft1d2((int64_t)nj, x, cj, iflag, eps, (int64_t)ms, fk, &opts);
  deinterleave(nj, cj, cj_re, cj_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft2d2")))
int nufft2d2_w(int nj, double *x, double *y,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms, int mt,
               double *fk_re, double *fk_im) {
  int nin = ms * mt;
  auto *fk = new CPX[nin];
  interleave(nin, fk_re, fk_im, fk);
  auto *cj = new CPX[nj];
  auto opts = default_opts();
  int ier = finufft2d2((int64_t)nj, x, y, cj, iflag, eps,
                       (int64_t)ms, (int64_t)mt, fk, &opts);
  deinterleave(nj, cj, cj_re, cj_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft3d2")))
int nufft3d2_w(int nj, double *x, double *y, double *z,
               double *cj_re, double *cj_im,
               int iflag, double eps, int ms, int mt, int mu,
               double *fk_re, double *fk_im) {
  int nin = ms * mt * mu;
  auto *fk = new CPX[nin];
  interleave(nin, fk_re, fk_im, fk);
  auto *cj = new CPX[nj];
  auto opts = default_opts();
  int ier = finufft3d2((int64_t)nj, x, y, z, cj, iflag, eps,
                       (int64_t)ms, (int64_t)mt, (int64_t)mu, fk, &opts);
  deinterleave(nj, cj, cj_re, cj_im);
  delete[] cj; delete[] fk;
  return ier;
}

// ── Type 3: nonuniform → nonuniform ─────────────────────────────────────────

__attribute__((export_name("nufft1d3")))
int nufft1d3_w(int nj, double *x,
               double *cj_re, double *cj_im,
               int iflag, double eps,
               int nk, double *s,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  auto *fk = new CPX[nk];
  auto opts = default_opts();
  int ier = finufft1d3((int64_t)nj, x, cj, iflag, eps,
                       (int64_t)nk, s, fk, &opts);
  deinterleave(nk, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft2d3")))
int nufft2d3_w(int nj, double *x, double *y,
               double *cj_re, double *cj_im,
               int iflag, double eps,
               int nk, double *s, double *t,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  auto *fk = new CPX[nk];
  auto opts = default_opts();
  int ier = finufft2d3((int64_t)nj, x, y, cj, iflag, eps,
                       (int64_t)nk, s, t, fk, &opts);
  deinterleave(nk, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

__attribute__((export_name("nufft3d3")))
int nufft3d3_w(int nj, double *x, double *y, double *z,
               double *cj_re, double *cj_im,
               int iflag, double eps,
               int nk, double *s, double *t, double *u,
               double *fk_re, double *fk_im) {
  auto *cj = new CPX[nj];
  interleave(nj, cj_re, cj_im, cj);
  auto *fk = new CPX[nk];
  auto opts = default_opts();
  int ier = finufft3d3((int64_t)nj, x, y, z, cj, iflag, eps,
                       (int64_t)nk, s, t, u, fk, &opts);
  deinterleave(nk, fk, fk_re, fk_im);
  delete[] cj; delete[] fk;
  return ier;
}

// ── Guru API ─────────────────────────────────────────────────────────────────

__attribute__((export_name("guru_makeplan")))
int guru_makeplan(int type, int dim, double *n_modes_d, int iflag,
                  int ntrans, double tol) {
  // Find a free slot
  int id = -1;
  for (int i = 0; i < 64; i++) {
    if (!plan_table_used[i]) { id = i; break; }
  }
  if (id < 0) return -1;

  int64_t n_modes[3] = {1, 1, 1};
  for (int i = 0; i < dim; i++)
    n_modes[i] = (int64_t)n_modes_d[i];

  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.nthreads = 1;
  opts.debug = 0;

  finufft_plan plan = nullptr;
  int ier = finufft_makeplan(type, dim, n_modes, iflag, ntrans, tol, &plan, &opts);
  if (ier != 0) return -1;

  plan_table[id].plan = plan;
  plan_table[id].type = type;
  plan_table[id].xj = plan_table[id].yj = plan_table[id].zj = nullptr;
  plan_table[id].s = plan_table[id].t = plan_table[id].u = nullptr;
  plan_table_used[id] = 1;
  return id;
}

static double *dup_array(double *src, int n) {
  if (!src || n <= 0) return nullptr;
  auto *dst = new double[n];
  std::memcpy(dst, src, n * sizeof(double));
  return dst;
}

static void free_plan_pts(PlanInfo &p) {
  delete[] p.xj; p.xj = nullptr;
  delete[] p.yj; p.yj = nullptr;
  delete[] p.zj; p.zj = nullptr;
  delete[] p.s;  p.s = nullptr;
  delete[] p.t;  p.t = nullptr;
  delete[] p.u;  p.u = nullptr;
}

__attribute__((export_name("guru_setpts")))
int guru_setpts(int handle, int nj, double *xj, double *yj, double *zj,
                int nk, double *s, double *t, double *u) {
  if (handle < 0 || handle >= 64 || !plan_table_used[handle]) return -1;
  PlanInfo &p = plan_table[handle];
  // Free any previously stored point arrays
  free_plan_pts(p);
  // Copy arrays so they persist after JS frees its WASM allocations
  p.xj = dup_array(xj, nj);
  p.yj = dup_array(yj, nj);
  p.zj = dup_array(zj, nj);
  p.s  = dup_array(s, nk);
  p.t  = dup_array(t, nk);
  p.u  = dup_array(u, nk);
  return finufft_setpts(p.plan,
                        (int64_t)nj, p.xj, p.yj, p.zj,
                        (int64_t)nk, p.s, p.t, p.u);
}

__attribute__((export_name("guru_execute")))
int guru_execute(int handle, double *in_re, double *in_im, int n_in,
                 double *out_re, double *out_im, int n_out) {
  if (handle < 0 || handle >= 64 || !plan_table_used[handle]) return -1;
  int type = plan_table[handle].type;

  if (type == 1 || type == 3) {
    // Input is strengths (n_in), output is coefficients/values (n_out)
    auto *cj = new CPX[n_in];
    interleave(n_in, in_re, in_im, cj);
    auto *fk = new CPX[n_out];
    int ier = finufft_execute(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, fk, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  } else {
    // Type 2: input is coefficients (n_in), output is values at targets (n_out)
    auto *fk = new CPX[n_in];
    interleave(n_in, in_re, in_im, fk);
    auto *cj = new CPX[n_out];
    int ier = finufft_execute(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, cj, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  }
}

__attribute__((export_name("guru_execute_adjoint")))
int guru_execute_adjoint(int handle, double *in_re, double *in_im, int n_in,
                         double *out_re, double *out_im, int n_out) {
  if (handle < 0 || handle >= 64 || !plan_table_used[handle]) return -1;
  int type = plan_table[handle].type;

  if (type == 1) {
    // Adjoint of type 1: input is coefficients (fk), output is values (cj)
    auto *fk = new CPX[n_in];
    interleave(n_in, in_re, in_im, fk);
    auto *cj = new CPX[n_out];
    int ier = finufft_execute_adjoint(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, cj, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  } else if (type == 2) {
    // Adjoint of type 2: input is strengths (cj), output is coefficients (fk)
    auto *cj = new CPX[n_in];
    interleave(n_in, in_re, in_im, cj);
    auto *fk = new CPX[n_out];
    int ier = finufft_execute_adjoint(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, fk, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  } else {
    // Adjoint of type 3: use finufft_execute with reversed data flow
    auto *fk = new CPX[n_in];
    interleave(n_in, in_re, in_im, fk);
    auto *cj = new CPX[n_out];
    int ier = finufft_execute(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, cj, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  }
}

__attribute__((export_name("guru_destroy")))
void guru_destroy(int handle) {
  if (handle < 0 || handle >= 64 || !plan_table_used[handle]) return;
  finufft_destroy(plan_table[handle].plan);
  plan_table[handle].plan = nullptr;
  free_plan_pts(plan_table[handle]);
  plan_table_used[handle] = 0;
}

} // extern "C"
