#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>   // C++
#include "../src/utils.h"

#define BIGINT long long

double rand01() {
    return (rand()%RAND_MAX)*1.0/RAND_MAX;
}

int main(int argc, char* argv[])
/* C complex type version speed test for squaring a big 1D array.

 g++ complexsquaretiming.cpp ../src/utils.o -o complexsquaretiming -O3
 ./complexsquaretiming

*/
{
  BIGINT M=1e8;
  CNTime timer;

  // C-type double
  double *x = (double *)malloc(sizeof(double)*M);
  for(BIGINT i=0;i<M;++i) x[i] = rand01();
  timer.start();
  for(BIGINT i=0;i<M;++i)     x[i] *= x[i];
  double t=timer.elapsedsec();
  printf("%lld C-type double squares in %.3g s\n",M,t);
  free(x);

  // C++ vector double
  std::vector<double> y(M);
  for(BIGINT i=0;i<M;++i) y[i] = rand01();
  timer.restart();
  for(BIGINT i=0;i<M;++i)     y[i] *= y[i];
  t=timer.elapsedsec();
  printf("%lld C++ std vector double squares in %.3g s\n",M,t);
  y.clear();

  // C-type double faking
  x = (double *)malloc(sizeof(double)*M);
  double *xi = (double *)malloc(sizeof(double)*M);
  for(BIGINT i=0;i<M;++i) { x[i] = rand01(); xi[i] = rand01();}
  timer.restart();
  for(BIGINT i=0;i<M;++i) {
    double r = x[i]*x[i] - xi[i]*xi[i];
    double j = 2*x[i]*xi[i];
    x[i] = r;
    xi[i] = j;
  }
  t=timer.elapsedsec();
  printf("%lld C-type doubles faking complex squares in %.3g s\n",M,t);
  free(x); free(xi);

  // C-type complex
  complex double *z = (complex double *)malloc(sizeof(complex double)*M);
  for(BIGINT i=0;i<M;++i)   z[i] = rand01() + I*rand01();
  timer.restart();
  for(BIGINT i=0;i<M;++i) {
    z[i] *= z[i];
  }
  t=timer.elapsedsec();
  printf("%lld C-type complex squares in %.3g s\n",M,t);
  free(z);

  return 0;
}
