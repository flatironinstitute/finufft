#include <finufft/test_defs.h>

/* Test that execute_adjoint applies the adjoint of execute, in the
   guru interface, for all types and dimensions. type 1 & 2 for now.
   We use the test_defs macros, as with other C-interface tests.
   Barnett 7/22/25

   Subtlety is that adjointness is subject to round-off, which is amplified
   by the r_max dynamic range, per dimension. That can get bad for USF=1.25.
*/

using namespace std;

int main() {

  BIGINT Ns[3] = {500, 40, 8}; // modes per dim, smallish probs
  BIGINT M     = 100000;       // NU pts, smallish, but enough so rand err small
  int isign    = +1;
  int ntr      = 1;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  // opts.upsampfac = 1.25;    // for by hand tests
#ifdef SINGLE
  FLT tol        = 1e-5; // requested transform tol (adj err worse near epsmach)
  FLT allowederr = 1e-4; // 1e3*epsmach due to usf=1.25 growth in round-off
  string name    = "adjointnessf";
#else
  FLT tol        = 1e-12; // requested transform tol (adj err worse near epsmach)
  FLT allowederr = 1e-10; // 1e6*epsmach, poss usf=1.25 growth in round-off
  string name    = "adjointness";
#endif

  cout << "adjointness: making random data...";
  // generate random non-uniform points on (x,y) and complex strengths (c)
  vector<FLT> x(M), y(M), z(M);
  vector<CPX> c(M);
  for (int i = 0; i < M; i++) {
    x[i] = PI * randm11(); // unif random in [-pi, pi)
    y[i] = PI * randm11();
    z[i] = PI * randm11();
    c[i] = crandm11(); // complex unif rand
  }

  // generate random mode coeffs (f), enough for any dim up to 3
  BIGINT Nmax = Ns[0] * Ns[1] * Ns[2];
  vector<CPX> f(Nmax);
  for (int i = 0; i < Nmax; i++) f[i] = crandm11();
  cout << " done" << endl;

  // allocate output arrays for adjoint testing (Capital denotes output)
  vector<CPX> C(M);
  vector<CPX> F(Nmax);
  FLT errmax = 0.0; // track worst errors across tests
  int ier, iermax = 0;

  for (int dim = 1; dim <= 3; ++dim) { // ....... loop over dims
    BIGINT N = 1;                      // compute actual num modes in this dim
    for (int d = 0; d < dim; ++d) N *= Ns[d];
    cout << "  dim=" << dim << ", M=" << M << " pts, N=" << N << " modes :" << endl;

    for (int type = 1; type <= 2; ++type) { // .......... loop over types
      cout << "\t" << type << " : ";
      FINUFFT_PLAN plan;
      FINUFFT_MAKEPLAN(type, dim, Ns, isign, ntr, tol, &plan, &opts);
      // no t3 yet...
      FINUFFT_SETPTS(plan, M, x.data(), y.data(), z.data(), 0, NULL, NULL, NULL);
      if (type == 1) {
        ier = FINUFFT_EXECUTE(plan, c.data(), F.data());         // c->F
        ier = FINUFFT_EXECUTE_ADJOINT(plan, C.data(), f.data()); // f->C
      } else if (type == 2) {                                    // reversed data flow
        ier = FINUFFT_EXECUTE(plan, C.data(), f.data());         // f->C
        ier = FINUFFT_EXECUTE_ADJOINT(plan, c.data(), F.data()); // c->F
      } else
        cout << name << ": type 3 not yet supported!" << endl;
      if (ier > 0) cout << "failure: ier=" << ier << endl;
      iermax = max(ier, iermax); // track if something failed
      FINUFFT_DESTROY(plan);

      // measure scalar error (f,F) - (C,c), should vanish by adjointness
      CPX ipc = 0.0, ipf = 0.0; // results for (C,c) and (f,F)
      for (int i = 0; i < M; i++) ipc += conj(C[i]) * c[i];
      for (int j = 0; j < N; j++) ipf += conj(f[j]) * F[j];
      FLT err = abs(ipc - ipf) / abs(ipc); // error rel to innerprod itself
      cout << " adj rel err " << err << endl;
      errmax = max(err, errmax);
    }
  }

  // report and exit code
  if (errmax > allowederr || iermax > 0) {
    cout << name << " failed! (allowederr=" << allowederr << ")" << endl;
    return 1;
  } else {
    cout << name << " passed" << endl;
    return 0;
  }
}
