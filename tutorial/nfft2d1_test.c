/* Minimal example of 2D "adjoint" (aka type 1) NUFFT using NFFT3 library,
   with user-given data, single-threaded, plus timer.  Barnett 5/17/24
   To compile (assuming nfft3 installed):
   gcc nfft2d1_test.c -o nfft2d1_test -lnfft3 -lfftw3 -lm
 */
#include "nfft3.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
  int N[2] = {300, 200}; // N1, N2 output mode numbers
  int M    = 500000;     // num. nonuniform input points
  int dim  = 2;
  nfft_plan p;
  nfft_init(&p, dim, N, M); // allocates user I/O arrays too
  // make some "user data" (we must use arrays that nfft allocated)...
  srand(0);                     // fix seed
  for (int j = 0; j < M; ++j) { // nonequispaced pts, and strengths f...
    p.x[2 * j]     = (double)rand() / RAND_MAX - 0.5; // x unif rand in [-1/2,1/2)
    p.x[2 * j + 1] = (double)rand() / RAND_MAX - 0.5; // y "
    p.f[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }

  clock_t before = clock();

  if (p.flags & PRE_ONE_PSI) // precompute psi, the entries of the matrix B
    nfft_precompute_one_psi(&p);
  nfft_adjoint(&p);          // do transform, write to p.f_hat

  double secs = (clock() - before) / (double)CLOCKS_PER_SEC;

  int kx = -17, ky = 33; // check one output f_hat(kx,ky) vs direct computation
  int kxout = kx + N[0] / 2;
  int kyout = ky + N[1] / 2;
  int i     = kyout + kxout * N[1]; // output index: array ordered x slow, y fast
  double complex f_hat_test = 0.0 + 0.0 * I;
  for (int j = 0; j < M; ++j)       // 2pi fac; p.x array is x interleaved with y...
    f_hat_test += p.f[j] * cexp(2 * M_PI * I *
                                ((double)kx * p.x[2 * j] + (double)ky * p.x[2 * j + 1]));
  double err = cabs(p.f_hat[i] - f_hat_test) / cabs(f_hat_test);
  printf("2D type 1 (NFFT3) done in %.3g s: f_hat[%d,%d]=%.12g+%.12gi, rel err %.3g\n",
         secs, kx, ky, creal(p.f_hat[i]), cimag(p.f_hat[i]), err);

  nfft_finalize(&p); // free the plan
  return 0;
}
