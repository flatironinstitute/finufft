#include "besseli.h"
#include <math.h>
#include "../src/utils.h"

int main(int argc, char* argv[])
// speed and accuracy tests for I_0 modified Bessel of 1st kind in besseli.cpp.
// Barnett 2/2/17
{
  (void)argc; //tell compiler this variable is unused
  (void)argv; //tell compiler this variable is unused
  double R=100.0;    // upper lim
  int n=1e7;
   CNTime timer; timer.start();
  for (int i=0;i<n;++i) {
    double x = R * (double)i/n;
    double y = besseli0(x);
  }
  printf("orig I0:               %.3g evals/sec\n",n/timer.elapsedsec());
  timer.restart();
  for (int i=0;i<n;++i) {
    double x = R * (double)i/n;
    double y = besseli0_approx(x);
  }
  printf("reduced cheby lengths: %.3g evals/sec\n",n/timer.elapsedsec());

  n=1e5;  // accuracy test
  double maxerr = 0.0, argmax = NAN;
  for (int i=0;i<n;++i) {
    double x = R * (double)i/n;
    double y = besseli0(x);
    double relerr = abs(y - besseli0_approx(x))/y;
    if (relerr>maxerr) { maxerr = relerr; argmax = x; }
  }
  printf("mex err over range [0,%g] : %.3g at x=%.3g\n",R,maxerr,argmax);
  return 0;
}
