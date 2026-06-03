// Minimal downstream consumer of an installed FINUFFT.
//
// Including <finufft.h> exercises the full public header chain
// (finufft_opts.h, finufft_errors.h, finufft/finufft_eitherprec.h and their
// transitive <finufft_common/defines.h>), so a missing installed header is a
// hard compile error here. Calling finufft_default_opts links against the
// installed library, exercising find_dependency() and the DLL export path.
#include <finufft.h>

#include <cstdio>

int main() {
  finufft_opts opts;
  finufft_default_opts(&opts);

  // A non-trivial round trip: a tiny type-1 1D transform.
  const int64_t M = 8;
  const int64_t N = 16;
  double x[M];
  std::complex<double> c[M];
  std::complex<double> fk[N];
  for (int64_t j = 0; j < M; ++j) {
    x[j] = 0.0;
    c[j] = {1.0, 0.0};
  }
  const int ier = finufft1d1(M, x, c, +1, 1e-6, N, fk, &opts);
  if (ier != 0) {
    std::printf("finufft1d1 returned error %d\n", ier);
    return 1;
  }
  std::printf("finufft consume OK\n");
  return 0;
}
