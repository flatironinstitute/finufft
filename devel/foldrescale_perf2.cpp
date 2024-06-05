/* performance comparison of fold-rescale macro vs function, for spreadinterp.
   Version 2: random array that cannot be branch-predicted in fold conditional.
   Also does binning (for N<1e4 this is const time, for >1e5 mem access slows).

   Compile with, eg on linux, double-prec:

   g++ -O3 -funroll-loops -march=native -I../include -fopenmp foldrescale_perf2.cpp -o
foldrescale_perf2 -lgomp g++ -O3 -funroll-loops -march=native -I../include -fopenmp
foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp -Ofast -fno-finite-math-only

   Flags: -DSINGLE for single-prec. OMP only used for random # gen.
          -DNOBIN to skip the binning, leaving just fold&rescale.
   Usage:
   ./foldrescale_perf [M [N]]
   M = number of calls to rescale/fold, N = grid size (not known at compile time)

   Examples
   ./foldrescale_perf2 1e8
   ./foldrescale_perf2 1e8 1e6

   Barnett 7/15/20


BETTER i7 GCC9 RESULTS:  (run ./foldrescale.sh)

BINNING (closer to spreadinterp application):

alex@fiona /home/alex/numerics/finufft/devel> g++-9 -O3 -funroll-loops -march=native
-I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp alex@fiona
/home/alex/numerics/finufft/devel> ./foldrescale_perf2 simple array sum:           	1.9
ns/call	(sum:540.8833119415621) simple bin over [-3pi,3pi):  	1.1 ns/call	(ans:100667) w/
RESCALE1 macro:       	4.3 ns/call	(sum:499894508.4253364) w/ RESCALE macro (pir=0):	6.7
ns/call	(sum:499894508.4253364) w/ RESCALE macro (pir=1):	4.5 ns/call
(sum:499894508.4253364) w/ foldrescale1:           	8.3 ns/call	(sum:499894508.4253364) w/
foldrescale2:           	7.0 ns/call	(sum:499894508.4253364) w/
foldrescale3:           	7.0 ns/call	(sum:499894508.4253364) w/ foldrescale
(pir=0):  	6.7 ns/call	(sum:499894508.4253364) w/ foldrescale (pir=1):  	8.2 ns/call
(sum:499894508.4253364) (ans:905754)

alex@fiona /home/alex/numerics/finufft/devel> g++-9 -O3 -funroll-loops -march=native
-I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp -Ofast
-fno-finite-math-only alex@fiona /home/alex/numerics/finufft/devel> ./foldrescale_perf2
simple array sum:           	0.4 ns/call	(sum:-9554.451222028649)
simple bin over [-3pi,3pi):  	1.5 ns/call	(ans:100815)
w/ RESCALE1 macro:       	2.0 ns/call	(sum:499919136.1859143)
w/ RESCALE macro (pir=0):	6.7 ns/call	(sum:499919136.1859143)
w/ RESCALE macro (pir=1):	1.9 ns/call	(sum:499919136.1859144)
w/ foldrescale1:           	7.8 ns/call	(sum:499919136.1859144)
w/ foldrescale2:           	6.7 ns/call	(sum:499919136.1859144)
w/ foldrescale3:           	7.0 ns/call	(sum:499919136.1859144)
w/ foldrescale (pir=0):  	6.4 ns/call	(sum:499919136.1859144)
w/ foldrescale (pir=1):  	8.1 ns/call	(sum:499919136.1859143)
            (ans:904913)
NOBIN:

alex@fiona /home/alex/numerics/finufft/devel> g++-9 -O3 -funroll-loops -march=native
-I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp -DNOBIN alex@fiona
/home/alex/numerics/finufft/devel> ./foldrescale_perf2 simple array sum:           	1.3
ns/call	(sum:-5028.023988434961) w/ RESCALE1 macro:       	1.3 ns/call
(sum:499984776.5128576) w/ RESCALE macro (pir=0):	6.4 ns/call	(sum:499984776.5128576) w/
RESCALE macro (pir=1):	1.4 ns/call	(sum:499984776.5128576) w/
foldrescale1:           	7.8 ns/call	(sum:499984776.5128576) w/
foldrescale2:           	6.2 ns/call	(sum:499984776.5128576) w/
foldrescale3:           	6.4 ns/call	(sum:499984776.5128576) w/ foldrescale
(pir=0):  	6.3 ns/call	(sum:499984776.5128576) w/ foldrescale (pir=1):  	8.2 ns/call
(sum:499984776.5128576) (ans:0)

alex@fiona /home/alex/numerics/finufft/devel> g++-9 -O3 -funroll-loops -march=native
-I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp -Ofast
-fno-finite-math-only -DNOBIN alex@fiona /home/alex/numerics/finufft/devel>
./foldrescale_perf2 simple array sum:           	0.4 ns/call	(sum:-14573.38274652959) w/
RESCALE1 macro:       	0.7 ns/call	(sum:499926457.4098142) w/ RESCALE macro (pir=0):	0.7
ns/call	(sum:499926457.4098142) w/ RESCALE macro (pir=1):	0.8 ns/call
(sum:499926457.4098142) w/ foldrescale1:           	1.0 ns/call	(sum:499926457.4098143) w/
foldrescale2:           	0.8 ns/call	(sum:499926457.4098142) w/ foldrescale3: 0.8 ns/call
(sum:499926457.4098142) w/ foldrescale (pir=0):  	0.9 ns/call	(sum:499926457.4098143) w/
foldrescale (pir=1):  	1.0 ns/call	(sum:499926457.4098144) (ans:0) Concl:
* foldrescale FUNCTION is only fast when Ofast & NOBIN, really weird.
* macro *is* faster than function, even modern g++.
* RESCALE is same as RESCALE1
* foldrescale (pir=0) is fastest of the FUNCS, but foldrescale2 is close,
  and beats foldrescale (pir=1) which would be the usual use case.




old i7 GCC RESULTS:  ./foldrescale.sh:  fast-math, then not

simple array sum:           	0.4 ns/call	(sum:81753.8964958474)
w/ RESCALE macro (pir=0):	0.8 ns/call	(sum:49992036081751.82)
w/ RESCALE macro (pir=1):	0.8 ns/call	(sum:99990169618500.31)
w/ foldrescale (pir=0):  	0.8 ns/call	(sum:49992036081751.82)
w/ foldrescale (pir=1):  	0.9 ns/call	(sum:99990169618500.31)

simple array sum:           	1.2 ns/call	(sum:75594.96014970534)
w/ RESCALE macro (pir=0):	4.8 ns/call	(sum:49993591075600.05)
w/ RESCALE macro (pir=1):	8.0 ns/call	(sum:99993530387176.88)
w/ foldrescale (pir=0):  	4.7 ns/call	(sum:49993591075600.05)
w/ foldrescale (pir=1):  	7.9 ns/call	(sum:99993530387176.88)

Oh dear!  fast-math really works here.

Issues w/ fast-math and isinf, isnan, etc:
https://stackoverflow.com/questions/22931147/stdisinf-does-not-work-with-ffast-math-how-to-check-for-infinity

can recover isnan handling with -Ofast -fno-finite-math-only     .. good!




*/

// since defs starts w/ dataTypes, FLT responds to -DSINGLE from compile line
#include "finufft/defs.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <vector>
// let's try the "modern" C++ way to time... yuk...
#include <chrono>
using namespace std::chrono;

// old coord-handling macro ------------------------------------------------
// #define RESCALE(x,N,p) (p ?                                           \
//                        (x*(FLT)M_1_2PI*N + (x*(FLT)M_1_2PI*N<-N/(FLT)2.0 ? (FLT)1.5 :
//                        (x*(FLT)M_1_2PI*N>N/(FLT)2.0 ? (FLT)-0.5 : (FLT)0.5))*N) : \
//                        (x<(FLT)0.0 ? x+(FLT)N : (x>(FLT)N ? x-(FLT)N : x)))
// casting makes no difference

// cleaner rewrite, no slower:
#define RESCALE(x, N, p)                                                    \
  (p ? (x + (x >= -PI ? (x < PI ? PI : -PI) : 3 * PI)) * ((FLT)M_1_2PI * N) \
     : (x >= 0.0 ? (x < (FLT)N ? x : x - (FLT)N) : x + (FLT)N))

// pirange=1 fixed ver of old coord-handling macro ------------------------
// #define RESCALE1(x,N) (x*(FLT)M_1_2PI*N + (x*(FLT)M_1_2PI*N<-N/(FLT)2.0 ? (FLT)1.5*N :
// (x*(FLT)M_1_2PI*N>N/(FLT)2.0 ? (FLT)-0.5*N : (FLT)0.5*N)))
// it does matter how written: this made faster...
// #define RESCALE1(x,N) (x*(FLT)M_1_2PI + (x*(FLT)M_1_2PI<-0.5 ? 1.5 :
// (x*(FLT)M_1_2PI>0.5 ? -0.5 : 0.5)))*N

#define RESCALE1(x, N) \
  (x + (x >= -PI ? (x < PI ? PI : -PI) : 3 * PI)) * ((FLT)M_1_2PI * N)

// function equivalents -----------------------------------------------------
static inline FLT foldrescale(FLT x, BIGINT N, int pirange)
// if pirange true, affine transform x so -pi maps to 0 and +pi to N. Then fold
// [-N,0) and [N,2N) back into [0,N), the range of the output.
// Replaces the RESCALE macro. Barnett 7/15/20.
{
  // affine rescale...
  FLT z = x;
  if (pirange) z = (N / (2 * PI)) * (x + PI); // PI is (FLT)M_PI in defs.h
  // fold...
  if (z < (FLT)0.0)
    z += (FLT)N;
  else if (z >= (FLT)N)
    z -= (FLT)N;
  return z;
}

static inline FLT foldrescale1(FLT x, BIGINT N)
// same as above but hardwired pirange=1. rescale then fold
{
  // affine rescale always...
  FLT z = (N / (2 * PI)) * (x + PI); // PI is (FLT)M_PI in defs.h
  // fold...
  if (z < (FLT)0.0)
    z += (FLT)N;
  else if (z >= (FLT)N)
    z -= (FLT)N;
  return z;
}

static inline FLT foldrescale2(FLT x, BIGINT N)
// same as above but hardwired pirange=1, flip so fold done before rescale
{
  if (x < -PI)
    x += 2 * PI;
  else if (x > PI)
    x -= 2 * PI;
  return (N / (2 * PI)) * (x + PI);
}

static inline FLT foldrescale3(FLT x, BIGINT N)
// same as above but hardwired pirange=1, flip so fold done before rescale
{
  if (x < -PI)
    x += 3 * PI;
  else if (x > PI)
    x -= PI;
  else
    x += PI;
  return (N / (2 * PI)) * x;
}

// ==========================================================================
int main(int argc, char *argv[]) {
  int M = 10000000; // default: # pts to test (>=1e7 is acc)
  int N = 100;      // grid size, matters that unknown @ compile

  if (argc > 1) {
    double w;
    sscanf(argv[1], "%lf", &w);
    M = (int)w;
  }
  if (argc > 2) {
    double w;
    sscanf(argv[2], "%lf", &w);
    N = (int)w;
  }
  std::vector<int> c(N, 0); // let's do basic binning while we're at it
                            // to prevent compiler optims
  int maxc = 0;             // use for max bin count

  // fill array w/ random #s (in par), deterministic seeds based on threads
  std::vector<FLT> x(M);
#pragma omp parallel
  {
    unsigned int s = omp_get_thread_num(); // needed for parallel random #s
#pragma omp for schedule(dynamic, 1000000)
    for (int i = 0; i < M; ++i)
      x[i] = 3.0 * PI * randm11r(&s); // unif over the folded domain
  }
  // (note when pirange=0 the conditional <0 vs >=0 still 1:2 random)
  // We'll reuse this array by rescaling/unrescaling by hand.

  FLT sum     = 0.0;
  auto tbegin = system_clock::now();
  for (int i = 0; i < M; ++i) sum += x[i];             // simply sweep through array
  duration<double> dur = system_clock::now() - tbegin; // dur.count() is sec
  printf("simple array sum:           \t%.1f ns/call\t(sum:%.16g)\n",
         1e9 * dur.count() / (double)M, sum);

#ifndef NOBIN
  tbegin = system_clock::now();
  for (int i = 0; i < M; ++i) {
    int b = (int)(N * ((1.0 / (6 * PI)) * x[i] + (FLT)0.5)); // in {0,..,N-1}
    ++c[b];
    // if (b<0 || b>=N) printf("b[%d]=%d (x=%.16g, flt
    // b=%.16g)\n",i,b,x[i],N*((1.0/(6*PI))*x[i] + 0.5));  // chk all indices ok!
  }
  dur = system_clock::now() - tbegin; // dur.count() is sec
  for (int b = 0; b < N; ++b)
    if (c[b] > maxc) maxc = c[b];     // somehow use it
  printf("simple bin over [-3pi,3pi):  \t%.1f ns/call\t(ans:%d)\n",
         1e9 * dur.count() / (double)M, maxc);
#endif

  sum    = 0.0; // hardwired pirange=1 MACRO.......................
  tbegin = system_clock::now();
  for (int i = 0; i < M; ++i) {
    FLT z = RESCALE1(x[i], N);
    sum += z;
#ifndef NOBIN
    ++c[(int)z]; // bin it
#endif
  }
  dur = system_clock::now() - tbegin; // dur.count() is sec
  printf("w/ RESCALE1 macro:       \t%.1f ns/call\t(sum:%.16g)\n",
         1e9 * dur.count() / (double)M, sum);

  for (int pirange = 0; pirange < 2; ++pirange) {
    if (!pirange)
      for (int i = 0; i < M; ++i) x[i] = (N / (2 * PI)) * (x[i] + PI); // rescale to [0,N)
    // FLT mx=0.0; for (int i=0;i<M;++i) if (x[i]>mx) mx=x[i];   // chk max
    // printf("max x=%.3g\n",mx);
    sum    = 0.0;
    tbegin = system_clock::now();
    for (int i = 0; i < M; ++i) {
      FLT z = RESCALE(x[i], N, pirange);
      sum += z;
#ifndef NOBIN
      ++c[(int)z]; // bin it
#endif
    }
    dur = system_clock::now() - tbegin; // dur.count() is sec
    printf("w/ RESCALE macro (pir=%d):\t%.1f ns/call\t(sum:%.16g)\n", pirange,
           1e9 * dur.count() / (double)M, sum);
    if (!pirange)
      for (int i = 0; i < M; ++i) x[i] = x[i] * ((2 * PI) / N) - PI; // undo rescale
  }

  sum    = 0.0; // hardwired pirange=1 FUNC.......................
  tbegin = system_clock::now();
  for (int i = 0; i < M; ++i) {
    FLT z = foldrescale1(x[i], N);
    sum += z;
#ifndef NOBIN
    ++c[(int)z]; // bin it
#endif
  }
  dur = system_clock::now() - tbegin; // dur.count() is sec
  printf("w/ foldrescale1:           \t%.1f ns/call\t(sum:%.16g)\n",
         1e9 * dur.count() / (double)M, sum);

  sum    = 0.0; // hardwired pirange=1 FUNC.......................
  tbegin = system_clock::now();
  for (int i = 0; i < M; ++i) {
    FLT z = foldrescale2(x[i], N);
    sum += z;
#ifndef NOBIN
    ++c[(int)z]; // bin it
#endif
  }
  dur = system_clock::now() - tbegin; // dur.count() is sec
  printf("w/ foldrescale2:           \t%.1f ns/call\t(sum:%.16g)\n",
         1e9 * dur.count() / (double)M, sum);

  sum    = 0.0; // hardwired pirange=1 FUNC.......................
  tbegin = system_clock::now();
  for (int i = 0; i < M; ++i) {
    FLT z = foldrescale3(x[i], N);
    sum += z;
#ifndef NOBIN
    ++c[(int)z]; // bin it
#endif
  }
  dur = system_clock::now() - tbegin; // dur.count() is sec
  printf("w/ foldrescale3:           \t%.1f ns/call\t(sum:%.16g)\n",
         1e9 * dur.count() / (double)M, sum);

  for (int pirange = 0; pirange < 2; ++pirange) {
    if (!pirange)
      for (int i = 0; i < M; ++i) x[i] = (N / (2 * PI)) * (x[i] + PI); // rescale to [0,N)
    sum    = 0.0;
    tbegin = system_clock::now();
    for (int i = 0; i < M; ++i) {
      FLT z = foldrescale(x[i], N, pirange);
      sum += z;
#ifndef NOBIN
      ++c[(int)z]; // bin it
#endif
    }
    dur = system_clock::now() - tbegin; // dur.count() is sec
    printf("w/ foldrescale (pir=%d):  \t%.1f ns/call\t(sum:%.16g)\n", pirange,
           1e9 * dur.count() / (double)M, sum);
    if (!pirange)
      for (int i = 0; i < M; ++i) x[i] = x[i] * ((2 * PI) / N) - PI; // undo rescale
  }

  // force it to not optimize away the bin filling steps:
  maxc = 0;
  for (int b = 0; b < N; ++b)
    if (c[b] > maxc) maxc = c[b]; // somehow use it
  printf("\t\t\t\t\t\t(ans:%d)\n", maxc);
  return 0;
}
