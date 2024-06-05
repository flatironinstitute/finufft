/* test piecewise polynomial eval of kernel, for accuracy, then speed, vs
   math exp(sqrt(..)) evals. Also writes to tmp file.

For dyn linked:
g++-9 test_ker_ppval.cpp -o test_ker_ppval -O3 -funroll-loops -march=native -fopenmp
For statically linked so can control glibc (avoid Matlab calling being different):
g++-9 test_ker_ppval.cpp -o test_ker_ppval -O3 -funroll-loops -march=native -fopenmp
-static -lmvec

For GCC vectorization info: -fopt-info

Usage: test_ker_ppval [M [w]]
where M is number of pts for the speed test, and w is kernel width
(accuracy should be around 10^{1-w} )

See also: gen_ker_horner_C_code.m, ker_ppval_coeff_mat.m, fig_speed_ker_ppval.m

Barnett 4/23/18 & later.


Results:

if delete sscanf for w, then can get 0.08s for Horner w/ g++-7,
(-funroll-loops doesn't matter), but rel err in sum goes to 1e-10 from 1e-14.

Without -fopenmp, plain eval is >2s (can't get the magic 0.2s to kick in).

g++ and g++-7 give different results with -Ofast (with -O3 they are identical).
I have even seen 1e-7 error for w=12 (which should be good to 1e-11)

Demo that sscanf for w can speed plain eval but slow some magic horner speed:

alex@fiona /home/alex/numerics/finufft/devel> g++-7 test_ker_ppval.cpp -o test_ker_ppval
-Ofast -march=native -funroll-loops -fopenmp WITHOUT SSCANF FOR w: alex@fiona
/home/alex/numerics/finufft/devel> ./test_ker_ppval 10000000 acc test: sup err scaled to
kernel peak of 1: 6.53e-11 exp(sqrt): M=10000000 w=12 in 1.03 s:	116 Meval/s
(ans=2.73717868002952e+19) Horner:    M=10000000 w=12 in 0.0812 s:	1.48e+03 Meval/s
(ans=2.73717867964406e+19) rel err in sum = 1.41e-10 WITH SSCANF FOR w: alex@fiona
/home/alex/numerics/finufft/devel> ./test_ker_ppval 10000000 acc test: sup err scaled to
kernel peak of 1: 6.53e-11 exp(sqrt): M=10000000 w=12 in 0.45 s:	267 Meval/s
(ans=2.73717867952762e+19) Horner:    M=10000000 w=12 in 0.483 s:	248 Meval/s
(ans=2.73717867952754e+19) rel err in sum = 3.01e-14

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

// Choose prec... (w=7 enough for single)
typedef double FLT;
// typedef float FLT;

static inline void evaluate_kernel_vector(FLT *ker, const FLT *args, const FLT beta,
                                          const FLT c, const int w)
/* Evaluate kernel for a vector of w arguments, must also be the int width par.
   The #pragra's need to be removed for icpc if -fopenmp not used; g++ is ok.
 */
{
#pragma omp simd
  for (int i = 0; i < w; i++) ker[i] = exp(beta * sqrt((FLT)1.0 - c * args[i] * args[i]));
    // gcc 5.4 can't simd the combined loop, hence we split the
    // out-of-support test to subsequent loop...
    // This check loop prevents getting 0.2s (600 Meval/s):
#pragma omp simd
  for (int i = 0; i < w; i++)
    if (fabs(args[i]) >= (FLT)w / 2) // note fabs not abs!
      ker[i] = 0.0;
}

static inline void kernel_vector_Horner(FLT *ker, FLT z, int w)
/* Evaluate kernel for a vector of N grid-spaced arguments offset by z/w from
   the standard kernel grid on [-1,1] the kernel support.
   See: gen_all_horner_C_code.m
*/
{
  // #include "../src/ker_horner_allw.c"
#include "../src/ker_horner_allw_loop.c"
}

int main(int argc, char *argv[]) {
  int M = (int)1e7;                        // # of reps (<2^31)
  if (argc > 1)
    sscanf(argv[1], "%d", &M);             // weirdly allows exp simd 10x faster, even on
                                           // gcc 5.4.0
  int w = 12;                              // spread width
  if (argc > 2) sscanf(argv[2], "%d", &w); // prevents the magic 0.2s, keeps at 0.4s
  FLT beta = 2.30 * w;                     // should match kernel params for acc test
  if (w == 2) beta = 2.20 * w;
  if (w == 3) beta = 2.26 * w;
  if (w == 4) beta = 2.38 * w;
  FLT c  = 4.0 / (FLT)(w * w);               // set up ker params for plain eval
  FLT iw = 1.0 / (FLT)w;                     // scale factor
  std::vector<FLT> x(w);
  std::vector<FLT> f(16), f2(16);            // length=MAX_NSPREAD

  int Macc   = 100;                          // test accuracy.......
  FLT superr = 0.0;
  for (int i = 0; i < Macc; ++i) {           // loop over eval grid sets
    FLT z = (2 * i) / (FLT)(Macc - 1) - 1.0; // local offset sweep through z in [-1,1]
    // printf("z=%g:\n",z);   // useful for calling w/ eg Macc=3
    kernel_vector_Horner(&f2[0], z, w); // eval kernel to f2, given offset z
    for (int j = 0; j < w; ++j)         // vector of args in [-w/2,w/2] ker supp
      x[j] = (-(FLT)w + 1.0 + z) / 2 + j;
    evaluate_kernel_vector(&f[0], &x[0], beta, c, w); // eval kernel into f
    for (int j = 0; j < w; ++j) {
      // printf("x=%.3g\tf=%.6g\tf2=%.6g\tf2-f=%.3g\n",x[j],f[j],f2[j],f2[j]-f[j]);
      FLT err = abs(f[j] - f2[j]);
      if (err > superr) superr = err;
    }
  }
  superr /= exp(beta);
  printf("acc test: sup err scaled to kernel peak of 1: %.3g\n", superr);

  // test speed...... plain eval
  clock_t start = clock();
  FLT ans       = 0.0;                    // dummy answer
  for (int i = 0; i < M; ++i) {           // loop over eval grid sets
    FLT xi = -w / 2.0 + i / (FLT)(M - 1); // offset in [-w/2, -w/2+1]
    for (int j = 0; j < w; ++j)
      x[j] = xi + (FLT)j;                 // vector of args for [-w/2,w/2] ker supp
    evaluate_kernel_vector(&f[0], &x[0], beta, c, w); // eval kernel into f
    for (int j = 0; j < w; ++j) {
      // printf("x=%.16g\tf=%.16g\n",x[j],f[j]);
      ans += f[j]; // do something cheap to use all f outputs
    }
  }
  double t = (double)(clock() - start) / CLOCKS_PER_SEC;
  printf("exp(sqrt): M=%d w=%d in %.3g s:\t%.3g Meval/s (ans=%.15g)\n", M, w, t,
         M * w / (t * 1.0e6), ans);

  // test speed...... Horner (on same set as above, so can check its sum)
  start    = clock();
  FLT ans2 = 0.0;                             // dummy answer
  for (int i = 0; i < M; ++i) {               // loop over eval grid sets
    FLT z = (2 * i) / (FLT)(M - 1) - 1.0;     // local offset sweep through z in [-1,1]
    kernel_vector_Horner(&f[0], z, w);        // eval kernel to f, given offset z
    for (int j = 0; j < w; ++j) ans2 += f[j]; // do something cheap to use all f outputs
  }
  double t2 = (double)(clock() - start) / CLOCKS_PER_SEC;
  printf("Horner:    M=%d w=%d in %.3g s:\t%.3g Meval/s (ans=%.15g)\n", M, w, t2,
         M * w / (t2 * 1.0e6), ans2);

  printf("rel err in sum = %.3g\n", fabs(ans - ans2) / fabs(ans)); // another acc

  // append timing data to tmp file...
  FILE *p = fopen("/tmp/test_ker_ppval.dat", "a");
  fprintf(p, "%d %d %.3f %.3f %.3g\n", M, w, t, t2, superr);
  fclose(p);

  return 0;
}
