#include <math.h>
//extern "C" { #include "complex.h" } //
#include <complex.h>  // fails w/ std=C++11
#include <stdlib.h>
#include <stdio.h>
#include <vector>   // C++
//#include <complex>  // C++  fails w/ complex.h
//#include <ccomplex>  // C++  fails
#include "../src/utils.h"

#define BIGINT long long

double rand01() {
    return (rand()%RAND_MAX)*1.0/RAND_MAX;
}

int main(int argc, char* argv[])
/* C complex type version speed test for mult two big 1D arrays.

 g++ complexmulttiming.cpp ../src/utils.o -o complexmulttiming -O3
 ./complexmulttiming

*/
{
  BIGINT M=1e8;
  CNTime timer;

  // C-type double
  double *x = (double *)malloc(sizeof(double)*M);
  double *x2 = (double *)malloc(sizeof(double)*M);
  for(BIGINT i=0;i<M;++i) { x[i] = rand01(); x2[i] = rand01(); }
  timer.start();
  for(BIGINT i=0;i<M;++i)     x[i] = x[i] * x2[i];
  double t=timer.elapsedsec();
  printf("%lld C-type double mults in %.3g s\n",M,t);
  free(x); free(x2);

  // C++ vector double
  std::vector<double> y(M),y2(M);
  for(BIGINT i=0;i<M;++i) { y[i] = rand01(); y2[i] = rand01(); }
  timer.restart();
  for(BIGINT i=0;i<M;++i)     y[i] = y[i] * y2[i];
  t=timer.elapsedsec();
  printf("%lld C++ std vector double mults in %.3g s\n",M,t);
  y.clear(); y2.clear();

  // C-type double complex by hand
  x = (double *)malloc(sizeof(double)*M); 
  x2 = (double *)malloc(sizeof(double)*M);
  double *xi = (double *)malloc(sizeof(double)*M);
  double *xi2 = (double *)malloc(sizeof(double)*M);
  for(BIGINT i=0;i<M;++i) xi[i] = rand01();
  timer.restart();
  for(BIGINT i=0;i<M;++i) {
    double r = x[i]*x2[i] - xi[i]*xi2[i];
    double j = x[i]*xi2[i] + xi[1]*x2[i];
    x[i] = r;
    xi[i] = j;
  }
  t=timer.elapsedsec();
  printf("%lld C-type doubles faking complex mults in %.3g s\n",M,t);
  free(x); free(xi); free(x2); free(xi2);

  // C-type complex
  complex double *z = (complex double *)malloc(sizeof(complex double)*M);
  complex double *z2 = (complex double *)malloc(sizeof(complex double)*M);
  for(BIGINT i=0;i<M;++i) {
    z[i] = rand01() + I*rand01();
    z2[i] = rand01() + I*rand01();
  }
  timer.restart();
  for(BIGINT i=0;i<M;++i) {
    z[i] = z[i] * z2[i];
  }
  t=timer.elapsedsec();
  printf("%lld C-type complex mults in %.3g s\n",M,t);
  free(z); free(z2);
  
  if (0) {  // fail
  // C++ complex vector  ... can't even get to compile
  //  std::vector<std::complex<double>> Z[M],Z2[M];
    for(BIGINT i=0;i<M;++i) {
      //     Z[i] = rand01() + I*rand01();
      //Z2[i] = rand01() + I*rand01();
  }
    timer.restart();
    for(BIGINT i=0;i<M;++i) {
      //  Z[i] = Z[i] * Z2[i];
    }
    t=timer.elapsedsec();
    printf("%lld C++ vector complex mults in %.3g s\n",M,t);
    //Z.clear(); Z2.clear();
  }

  return 0;
}
