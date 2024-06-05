/* Minimal example of 2D "adjoint" (aka type 1) NUFFT using FINUFFT
   to match NFFT3's conventions, and the same user-given data as
   for nfft2d1_test.c.   Single-threaded, timer.   Barnett 5/18/24
   To compile (assuming FINUFFT include and lib in path):
   gcc migrate2d1_test.c -o migrate2d1_test -lfinufft -lfftw3 -lm
 */
#include <complex.h>
#include <finufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
  int N[2]   = {300, 200}; // N0, N1 output shape in nfft3 sense
  int M      = 500000;     // num. nonuniform input points
  double tol = 1e-13;      // user must choose (unlike nfft3's simple call)

  // user allocates all external arrays (and no internal ones)
  double *x = (double *)malloc(sizeof(double) * M); // x (0th) coords only here
  double *y = (double *)malloc(sizeof(double) * M); // y (1st) coords need separate ptr
  double complex *f = (double complex *)malloc(sizeof(double complex) * M);
  double complex *f_hat =
      (double complex *)malloc(sizeof(double complex) * N[0] * N[1]); // output

  // start with exactly the same "user data" as in nfft2d1_test.c...
  srand(0);                                 // fix seed
  for (int j = 0; j < M; ++j) {             // nonequispaced pts, and strengths f...
    x[j] = (double)rand() / RAND_MAX - 0.5; // x unif rand in [-1/2,1/2)
    y[j] = (double)rand() / RAND_MAX - 0.5; // y "
    f[j] =
        2 * ((double)rand() / RAND_MAX) - 1 + I * (2 * ((double)rand() / RAND_MAX) - 1);
  }

  clock_t before = clock();

  // do transform, includes precompute, writing to f_hat...
  for (int j = 0; j < M; ++j) { // change user coords so finufft same as nfft3
    x[j] *= 2 * M_PI;
    y[j] *= 2 * M_PI;           // scales from 1-periodic to 2pi-periodic
  }
  finufft_opts opts;            // opts struct
  finufft_default_opts(&opts);  // set default opts (must start with this)
  opts.nthreads = 1;            // enforce single-thread
  int ier = finufft2d1(M, y, x, f, +1, tol, N[1], N[0], f_hat, &opts); // both x,y and
                                                                       // N0,N1 swapped!

  double secs = (clock() - before) / (double)CLOCKS_PER_SEC;

  // now test that f_hat is as it would have been if original data were sent to nfft3...
  int kx = -17, ky = 33; // check one output f_hat(kx,ky) vs direct computation
  int kxout = kx + N[0] / 2;
  int kyout = ky + N[1] / 2;
  int i     = kyout + kxout * N[1]; // the output index (nfft3 convention, not FINUFFT's)
  double complex f_hat_test = 0.0 + 0.0 * I;
  for (int j = 0; j < M; ++j)       // since x,y were mult by 2pi, no such factor here...
    f_hat_test += f[j] * cexp(I * ((double)kx * x[j] + (double)ky * y[j]));
  double err = cabs(f_hat[i] - f_hat_test) / cabs(f_hat_test);
  printf("2D type 1 (FINUFFT) in %.3g s: f_hat[%d,%d]=%.12g+%.12gi, rel err %.3g\n", secs,
         kx, ky, creal(f_hat[i]), cimag(f_hat[i]), err);

  free(x);
  free(y);
  free(f);
  free(f_hat); // user deallocates own I/O arrays
  return ier;
}
