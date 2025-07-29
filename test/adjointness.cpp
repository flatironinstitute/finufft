#include "utils/norms.hpp"
#include <finufft/test_defs.h>

/* Test that execute_adjoint applies the adjoint of execute, in the
   guru interface, for all types and dimensions.
   We use the test_defs macros, as with other C-interface tests.
   Barnett 7/23/25. More stable relative error denom, discussion 7/29/25.

   Test tolerances are hard-wired in this code, not via cmd line.
   Subtlety is that adjointness is subject to round-off, which is amplified
   by the r_dyn dynamic range, per dimension. That can get bad for USF=1.25,
   esp in type 3 (eg by default if eps~1e-8 in double prec).
   See discussions "Repeatability" and "Adjointness" in docs/trouble.rst

   Discussion of the "correct" denominator to claim "relative" err in (f,F)-(C,c):
   ||Ferr||_2/||F||_2 is our NUFFT metric, as per our SISC 2019 FINUFFT paper.
   Expect err in (f,F) ~ ||f||_2.(error in one entry of F), by rotational
   invariance in C^N (where N is really Nused the number of modes).
   And (err in one entry of F) ~ ||Ferr||_2 / sqrt(N). Combining with the metric
   above, explains our choice denom = ||f||.||F||/sqrt(N) to give a rel measure
   of adjointness error. Has much less fluctuation than old denom=(f,F) which it
   turned out had zero-mean Gaussian pdf (=> fat-tailed pdf of reciprocal, bad!)
*/

using namespace std;

int main() {

  BIGINT Ns[3] = {200, 40, 6}; // modes per dim, smallish probs (~1e5 max)
  BIGINT M     = 50000;        // NU pts, smallish, but enough so rand err small
  int isign    = +1;
  int ntr      = 1;            // how many transforms (one for now)
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  // opts.upsampfac = 1.25;    // experts use to override default USF
#ifdef SINGLE
  FLT tol        = 1e-6; // requested transform tol (small enough to force USF=2)
  FLT allowederr = 1e-4; // ~1e3*epsmach (allow USF=1.25 larger r_dyn)
  string name    = "adjointnessf";
#else
  FLT tol        = 1e-12; // requested transform tol (eps<=1e-9 => USF=2 guaranteed)
  FLT allowederr = 1e-10; // ~1e6*epsmach (USF=2 r_dyn^3<1e3, but allow USF=1.25)
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

  // generate random freq targs for type 3 only (t3 always uses Nmax targs)
  vector<FLT> s(Nmax), t(Nmax), u(Nmax);
  for (int i = 0; i < Nmax; i++) { // space-bandwidth prod O(Nd) for dim d
    s[i] = Ns[0] / 2 * randm11();  // unif random in [-N1/2,N1/2]
    t[i] = Ns[1] / 2 * randm11();
    u[i] = Ns[2] / 2 * randm11();
  }

  // allocate output arrays for adjoint testing (Capital denotes output)
  vector<CPX> C(M);
  vector<CPX> F(Nmax);
  FLT errmax = 0.0; // track worst errors across tests
  int ier, ieradj, iermax = 0;

  for (int dim = 1; dim <= 3; ++dim) { // ....... loop over dims
    BIGINT N = 1;                      // compute actual num modes in this dim
    for (int d = 0; d < dim; ++d) N *= Ns[d];
    cout << "  dim=" << dim << ", M=" << M << " pts, N=" << N << " modes:" << endl;

    for (int type = 1; type <= 3; ++type) { // .......... loop over types
      cout << "\ttype " << type << ": ";
      FINUFFT_PLAN plan;
      FINUFFT_MAKEPLAN(type, dim, Ns, isign, ntr, tol, &plan, &opts);
      // always input NU pts and freq targs (latter only used by t3)...
      FINUFFT_SETPTS(plan, M, x.data(), y.data(), z.data(), Nmax, s.data(), t.data(),
                     u.data());
      if (type != 2) {                                              // t1 or t3
        ier    = FINUFFT_EXECUTE(plan, c.data(), F.data());         // c->F
        ieradj = FINUFFT_EXECUTE_ADJOINT(plan, C.data(), f.data()); // f->C
      } else {                                              // has reversed data flow
        ier    = FINUFFT_EXECUTE(plan, C.data(), f.data()); // f->C
        ieradj = FINUFFT_EXECUTE_ADJOINT(plan, c.data(), F.data()); // c->F
      }
      if (ier > 0) cout << "\texecute failure: ier=" << ier << endl;
      if (ieradj > 0) cout << "\texecute_adjoint failure: ier=" << ieradj << endl;
      iermax = max(max(ier, ieradj), iermax); // track if something failed
      FINUFFT_DESTROY(plan);

      // measure scalar error (f,F) - (C,c), should vanish by adjointness...
      CPX ipc = 0.0, ipf = 0.0;           // inner-prod results for (C,c) and (f,F)
      for (int i = 0; i < M; i++) ipc += conj(C[i]) * c[i];
      int Nused = (type == 3) ? Nmax : N; // how many modes or freqs used
      for (int j = 0; j < Nused; j++) ipf += conj(f[j]) * F[j];

      // denominator for rel error (twonorm in utils.hpp), see discussion at top:
      // FLT denom = twonorm(M,C.data()) * twonorm(M,c.data()) / sqrt(M);  // v sim
      FLT denom = twonorm(Nused, F.data()) * twonorm(Nused, f.data()) / sqrt(Nused);
      FLT err   = abs(ipc - ipf) / denom;     // scale rel to norms of vectors in ipc
      cout << " adj rel err " << err << endl; // "\t denom=" << denom << endl;
      errmax = max(err, errmax);
    }
  }

  // report and exit code
  if (errmax > allowederr || iermax > 0) {
    cout << name << " failed! (allowederr=" << allowederr << ", iermax=" << iermax << ")"
         << endl;
    return 1;
  } else {
    cout << name << " passed" << endl;
    return 0;
  }
}
