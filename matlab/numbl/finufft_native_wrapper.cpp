// C++ wrapper around FINUFFT for native shared library (koffi FFI).
// Exports guru API with separated real/imaginary arrays.
// Statically links FINUFFT — no external dependencies at runtime.

#include <complex>
#include <cstdlib>
#include <cstring>
#include <finufft.h>

using CPX = std::complex<double>;

#define EXPORT extern "C" __attribute__((visibility("default")))

// ── Guru API: handle table ──────────────────────────────────────────────────

struct PlanInfo {
  finufft_plan plan;
  int type;
  double *xj, *yj, *zj, *s, *t, *u;
};

static PlanInfo plan_table[64];
static int plan_table_used[64] = {};

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

EXPORT int guru_makeplan(int type, int dim, double *n_modes_d, int iflag,
                         int ntrans, double tol) {
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

EXPORT int guru_setpts(int handle, int nj, double *xj, double *yj, double *zj,
                       int nk, double *s, double *t, double *u) {
  if (handle < 0 || handle >= 64 || !plan_table_used[handle]) return -1;
  PlanInfo &p = plan_table[handle];
  free_plan_pts(p);
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

EXPORT int guru_execute(int handle, double *in_re, double *in_im, int n_in,
                        double *out_re, double *out_im, int n_out) {
  if (handle < 0 || handle >= 64 || !plan_table_used[handle]) return -1;
  int type = plan_table[handle].type;

  if (type == 1 || type == 3) {
    auto *cj = new CPX[n_in];
    interleave(n_in, in_re, in_im, cj);
    auto *fk = new CPX[n_out];
    int ier = finufft_execute(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, fk, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  } else {
    auto *fk = new CPX[n_in];
    interleave(n_in, in_re, in_im, fk);
    auto *cj = new CPX[n_out];
    int ier = finufft_execute(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, cj, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  }
}

EXPORT int guru_execute_adjoint(int handle, double *in_re, double *in_im, int n_in,
                                double *out_re, double *out_im, int n_out) {
  if (handle < 0 || handle >= 64 || !plan_table_used[handle]) return -1;
  int type = plan_table[handle].type;

  if (type == 1) {
    auto *fk = new CPX[n_in];
    interleave(n_in, in_re, in_im, fk);
    auto *cj = new CPX[n_out];
    int ier = finufft_execute_adjoint(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, cj, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  } else if (type == 2) {
    auto *cj = new CPX[n_in];
    interleave(n_in, in_re, in_im, cj);
    auto *fk = new CPX[n_out];
    int ier = finufft_execute_adjoint(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, fk, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  } else {
    auto *fk = new CPX[n_in];
    interleave(n_in, in_re, in_im, fk);
    auto *cj = new CPX[n_out];
    int ier = finufft_execute(plan_table[handle].plan, cj, fk);
    deinterleave(n_out, cj, out_re, out_im);
    delete[] cj; delete[] fk;
    return ier;
  }
}

EXPORT void guru_destroy(int handle) {
  if (handle < 0 || handle >= 64 || !plan_table_used[handle]) return;
  finufft_destroy(plan_table[handle].plan);
  plan_table[handle].plan = nullptr;
  free_plan_pts(plan_table[handle]);
  plan_table_used[handle] = 0;
}
