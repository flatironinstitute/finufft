/* deterministic speed test for 2d2 interpolation, without wrapping effects.

   g++ time2d2interp.cpp -o time2d2interp -Ofast; OMP_NUM_THREADS=1 ./time2d2interp

   If ns=10 statically defined in code:
   xeon gcc 7.3 -Ofast -march=native:   0.53 s
   xeon gcc 7.3 -Ofast:                 0.44 s
   xeon gcc 7.3 -O3:                    0.88 s
   xeon gcc 7.3 -O2:                    1.2 s
   xeon gcc 7.3:                        8.6 s

   if ns=10 read from argv:
   xeon gcc 7.3 -Ofast:                 1.0 s
   xeon gcc 7.3 -Ofast -march=native:   1.4 s

   This models the dir=2 interp in (where we think interp_square needs 1.6 s):
   test/spreadtestnd 2 1e7 1e6 1e-9

   The latter should be run in the 2d2nowrap branch which has no-wrapping 2d2.

   Barnett 5/1/18
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

// Choose prec for floating pt...
typedef double FLT;

#define MAXNS 16

int main(int argc, char *argv[]) {
  int M = 10000000; // NU pts
  int n = 2000;     // U grid pts per dimension (needn't be huge)
  if (argc > 1) sscanf(argv[1], "%d", &M);
  if (argc > 2) sscanf(argv[2], "%d", &n);
  int ns = 10; // kernel width
  if (argc > 3) sscanf(argv[3], "%d", &ns);
  FLT ker1[MAXNS], ker2[MAXNS];

  std::vector<FLT> du(2 * n * n);     // U "input" array, with...
  for (int i = 0; i < 2 * n * n; ++i) // something in it
    du[i] = (FLT)i;

  clock_t start = clock();
  FLT tot[2]    = {0.0, 0.0};      // complex output total
  int N1 = n, N2 = n;
  int i1 = n / 4, i2 = n / 4 + 7;  // starting pt for bottom left coords of interp box

  for (int i = 0; i < M; ++i) {    // loop over NU pts ..............
    for (int j = 0; j < ns; ++j) { // some fixed 1d ker evals, dep on NU pt
      ker1[j] = 1.0 - 0.1 * (j - 4.7) * (j - 4.6) + ((FLT)i) * 1e-7;
      ;
      ker2[j] = 0.7 - 0.04 * (j - 3.7) * (j - 3.2) + ((FLT)i) * (-0.6e-7);
    }
    FLT out[2] = {0.0, 0.0}; // re,im for result for each NU pt

    // core loop of interp_square... (no wrapping)
    for (int dy = 0; dy < ns; dy++) {
      int j = N1 * (i2 + dy) + i1;
      for (int dx = 0; dx < ns; dx++) {
        FLT k = ker1[dx] * ker2[dy];
        out[0] += du[2 * j] * k;
        out[1] += du[2 * j + 1] * k;
        ++j;
      }
    }
    // printf("i=%d i1=%d i2=%d out=(%g,%g)\n",i,i1,i2,out[0],out[1]);

    tot[0] += out[0]; // do something w/ answers
    tot[1] += out[1];
    i1 += 1;          // slowly(!) advance the box corner up and across the grid
    // (since N,M same order, sweeps O(1) times across the U grid, as bin sort)
    if (i1 > 3 * n / 4) {
      i1 -= n / 2;
      i2 += 1;
    } // keep spread box away from edges
    // i2 += 57;                // move far in slow direc - causes pain
    if (i2 > 3 * n / 4) i2 -= n / 2;

  } // .......................
  double t = (double)(clock() - start) / CLOCKS_PER_SEC;
  printf("M=%d from N=%d^2, ns=%d: tot[0]=%.15g \t%.3g s\n", M, n, ns, tot[0], t);
  printf("%.3g spread pts/s\n", M * ns * ns / t);
  return 0;
}
