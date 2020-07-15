/* performance comparison of fold-rescale macro vs function, for spreadinterp.
   Version 2: random array that cannot be branch-predicted in fold conditional.

   Compile with, eg on linux, double-prec:

   g++-9 -O3 -funroll-loops -march=native -I../include -fopenmp foldrescale_perf2.cpp -o foldrescale_perf2 -lgomp
   Use -DSINGLE for single-prec. OMP only used for random # gen.

   Usage:
   ./foldrescale_perf [M [N]]
   M = number of calls to rescale/fold, N = grid size (not known at compile time)

   Examples
   ./foldrescale_perf2 1e8
   ./foldrescale_perf2 1e8 1e6

   Barnett 7/15/20


i7 GCC RESULTS:  ./foldrescale.sh:  fast-math, then not

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

Oh dear!
*/

// since defs starts w/ dataTypes, FLT responds to -DSINGLE from compile line
#include "defs.h"

#include <math.h>
#include <stdio.h>
#include <vector>
#include <omp.h>
// let's try the "modern" C++ way to time... yuk...
#include <chrono>
using namespace std::chrono;

// old coord-handling macro ------------------------------------------------
#define RESCALE(x,N,p) (p ?                                           \
                        (x*(FLT)M_1_2PI*N + (x*(FLT)M_1_2PI*N<-N/(FLT)2.0 ? (FLT)1.5 : (x*(FLT)M_1_2PI*N>N/(FLT)2.0 ? (FLT)-0.5 : (FLT)0.5))*N) : \
                        (x<(FLT)0.0 ? x+(FLT)N : (x>(FLT)N ? x-(FLT)N : x)))
// casting makes no difference
//		     (x<0 ? x+N : (x>N ? x-N : x)))


// function equivalent -----------------------------------------------------
FLT foldrescale(FLT x, BIGINT N, int pirange)
// if pirange true, affine transform x so -pi maps to 0 and +pi to N. Then fold
// [-N,0) and [N,2N) back into [0,N), the range of the output.
// Replaces the RESCALE macro. Barnett 7/15/20.
{
  // affine rescale...
  FLT z = x;
  if (pirange)
    z = (N/(2*PI)) * (x+PI);                  // PI is (FLT)M_PI in defs.h
  // fold...
  if (z<(FLT)0.0)
    z += (FLT)N;
  else if (z>=(FLT)N)
    z -= (FLT)N;
  return z;
} 

// ==========================================================================
int main(int argc, char* argv[])
{
  int M=10000000;                 // default: # pts to test (>=1e7 is acc)
  long int N = 1000000;           // grid size, matters that unknown @ compile
  
  if (argc>1) { double w; sscanf(argv[1],"%lf",&w); M = (int)w; }
  if (argc>2) { double w; sscanf(argv[2],"%lf",&w); N = (long int)w; }  

  // fill array w/ random #s (in par), deterministic seeds based on threads
  std::vector<FLT> x(M);
#pragma omp parallel
  {
    unsigned int s=omp_get_thread_num();  // needed for parallel random #s
#pragma omp for schedule(dynamic,1000000)
    for (int i=0; i<M; ++i)
      x[i] = 3.0*PI*randm11r(&s);          // unif over the folded domain
  }
  // (note when pirange=0 the conditional <0 vs >=0 still 1:2 random)
  // We'll reuse this array by rescaling/unrescaling by hand.
  
  FLT sum=0.0;
  auto tbegin = system_clock::now();
  for (int i=0;i<M;++i)
    sum += x[i];                          // simply sweep through array
  duration<double> dur = system_clock::now() - tbegin;   // dur.count() is sec
  printf("simple array sum:           \t%.1f ns/call\t(sum:%.16g)\n",1e9*dur.count()/(double)M,sum);

  for (int pirange=0;pirange<2;++pirange) {
    if (!pirange)
      for (int i=0;i<M;++i) x[i] = (N/(2*PI)) * (x[i]+PI);   // rescale to [0,N)
    //FLT mx=0.0; for (int i=0;i<M;++i) if (x[i]>mx) mx=x[i];   // chk max
    //printf("max x=%.3g\n",mx);
    sum = 0.0;
    tbegin = system_clock::now();
    for (int i=0;i<M;++i)
      sum += RESCALE(x[i],N,pirange);
    dur = system_clock::now() - tbegin;   // dur.count() is sec
    printf("w/ RESCALE macro (pir=%d):\t%.1f ns/call\t(sum:%.16g)\n",pirange,1e9*dur.count()/(double)M,sum);
    if (!pirange)
      for (int i=0;i<M;++i) x[i] = x[i]*((2*PI)/N) - PI;   // undo rescale
  }
  
  for (int pirange=0;pirange<2;++pirange) {
    if (!pirange)
      for (int i=0;i<M;++i) x[i] = (N/(2*PI)) * (x[i]+PI);   // rescale to [0,N)
    sum = 0.0;
    tbegin = system_clock::now();
    for (int i=0;i<M;++i)
      sum += foldrescale(x[i],N,pirange);
    dur = system_clock::now() - tbegin;   // dur.count() is sec
    printf("w/ foldrescale (pir=%d):  \t%.1f ns/call\t(sum:%.16g)\n",pirange,1e9*dur.count()/(double)M,sum);
    if (!pirange)
      for (int i=0;i<M;++i) x[i] = x[i]*((2*PI)/N) - PI;   // undo rescale
  }
  
  return 0;
}
