/* performance comparison of fold-rescale macro vs function, for spreadinterp.
   Version 1: deterministic x values that are easy to branch-predict.
              Indeed, we see no cost to the folding conditional.

   Compile with, eg on linux, double-prec:

   g++ -O3 -funroll-loops -march=native -I../include foldrescale_perf.cpp -o
   foldrescale_perf

   Use -DSINGLE for single-prec

   Usage:
   ./foldrescale_perf [M [N]]
   M = number of calls to rescale/fold; N = folding integer

   Examples
   ./foldrescale_perf
   ./foldrescale_perf 1e9
   ./foldrescale_perf 1e9 1e6

   Barnett 7/15/20
*/

#include "finufft/defs.h"
#include <math.h>
#include <stdio.h>
// let's try the "modern" C++ way to time... yuk...
#include <chrono>
using namespace std::chrono;

// Choose prec from compile line
#ifdef SINGLE
#define FLT float
#else
#define FLT double
#endif

// old coord-handling macro ------------------------------------------------
#define RESCALE(x, N, p)                                                         \
  (p ? (x * (FLT)M_1_2PI * N +                                                   \
        (x * (FLT)M_1_2PI * N < -N / (FLT)2.0                                    \
             ? (FLT)1.5                                                          \
             : (x * (FLT)M_1_2PI * N > N / (FLT)2.0 ? (FLT) - 0.5 : (FLT)0.5)) * \
            N)                                                                   \
     : (x < 0 ? x + N : (x > N ? x - N : x)))

// function equivalent -----------------------------------------------------
FLT foldrescale(FLT x, BIGINT N, int pirange)
// if pirange true, affine transform x so -pi maps to 0 and +pi to N. Then fold
// [-N,0) and [N,2N) back into [0,N), the range of the output.
// Replaces the RESCALE macro. Barnett 7/15/20.
{
  // affine rescale...
  FLT z = x;
  if (pirange)
    z = (N / (2 * PI)) * (x + PI); // PI is (FLT)M_PI in defs.h
  else
    z = x;
  // fold...
  if (z < (FLT)0.0)
    z += (FLT)N;
  else if (z >= (FLT)N)
    z -= (FLT)N;
  return z;
}

// ==========================================================================
int main(int argc, char *argv[]) {
  int M      = 100000000; // default: # pts to test
  long int N = 1000000;   // default: grid size, doesn't matter

  if (argc > 1) {
    double w;
    sscanf(argv[1], "%lf", &w);
    M = (int)w;
  }
  if (argc > 2) {
    double w;
    sscanf(argv[2], "%lf", &w);
    N = (long int)w;
  }

  FLT sum     = 0.0;
  auto tbegin = system_clock::now();
  for (int i = 0; i < M; ++i) {                 // v predictable x values,
    FLT x = (FLT)(-10.0) + i * ((FLT)20.0 / N); // I hope cheap; let's see!
    sum += x;
  }
  duration<double> dur = system_clock::now() - tbegin; // dur.count() is sec
  printf("backgnd ops:              \t%.3g s/call\t\t(sum:%.12g)\n", dur.count() / M,
         sum);

  sum = 0.0;
  for (int pirange = 0; pirange < 2; ++pirange) {
    tbegin = system_clock::now();
    for (int i = 0; i < M; ++i) {
      FLT x = (FLT)(-10.0) + i * ((FLT)20.0 / N);
      FLT z = RESCALE(x, N, pirange);
      sum += z;
    }
    dur = system_clock::now() - tbegin; // dur.count() is sec
    printf("w/ RESCALE macro (pir=%d):\t%.3g s/call\t\t(sum:%.12g)\n", pirange,
           dur.count() / M, sum);
  }

  sum = 0.0;
  for (int pirange = 0; pirange < 2; ++pirange) {
    tbegin = system_clock::now();
    for (int i = 0; i < M; ++i) {
      FLT x = (FLT)(-10.0) + i * ((FLT)20.0 / N);
      FLT z = foldrescale(x, N, pirange);
      sum += z;
    }
    dur = system_clock::now() - tbegin; // dur.count() is sec
    printf("w/ foldrescale (pir=%d):  \t%.3g s/call\t\t(sum:%.12g)\n", pirange,
           dur.count() / M, sum);
  }

  return 0;
}
