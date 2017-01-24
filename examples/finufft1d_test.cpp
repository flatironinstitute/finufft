#include <math.h>
#include "../src/utils.h"
#include "../src/finufft1d.h"
#include <stdio.h>

// C++ stuff
#include <iostream>
#include <iomanip>
#include <vector>
using namespace std;

#include <complex>          // C++ type complex
#define dcomplex complex<double>
#define ima complex<double>{0.0,1.0}


int main(int argc, char* argv[])
/* Test executable for nufft1d.

   Example: finufft1d_test 1000000 1000000 1e-12

 All complex arith done by hand for now. Barnett 1/22/17
*/
{
  BIGINT M = 1e6, N = 1e6;    // defaults: M = # srcs, N = # modes out
  double tol = 1e-6;          // default
  int isign = +1;
  if (argc>1)
    sscanf(argv[1],"%ld",&N);    // todo: what if BIGINT = long long ?
  if (argc>2)
    sscanf(argv[2],"%ld",&M);
  if (argc>3) {
    sscanf(argv[3],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>4) {
    fprintf(stderr,"Usage: finufft1d_test [Nmodes [Nsrc [tol]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);

  double *x = (double *)malloc(sizeof(double)*M);     // NU pts
  dcomplex* c = (dcomplex*)malloc(sizeof(dcomplex)*M);   // strengths 
  for (BIGINT j=0; j<M; ++j) x[j] = M_PI*(2*rand01()-1);
  for (BIGINT j=0; j<M; ++j) c[j] = 2*rand01()-1 + ima*(2*rand01()-1);
  dcomplex* F = (dcomplex*)malloc(sizeof(dcomplex)*N);   // output
  CNTime timer; timer.start();
  int ier = finufft1d1(M,x,(double*)c,isign,tol,N,(double*)F);
  //for (int j=0;j<N;++j) cout<<F[j]<<endl;
  double t=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    return 1;
  } else
    printf("\t%ld NU pts to %ld modes in %.3g s \t%.3g NU pts/s\n",M,N,t,M/t);

  BIGINT nt = N/2 - 7;   // compare direct eval of this mode
  dcomplex Ft = {0.0,0.0};
  for (BIGINT j=0; j<M; ++j)
    Ft += c[j] * exp(ima*((double)(isign*nt))*x[j]);
  Ft /= M;
  //cout << Ft << endl << F[N/2+nt] << endl;
  printf("rel err in F[%ld]: %.3g\n",nt,abs(1.0-F[N/2+nt]/Ft));

  free(x); free(c); free(F);
  return ier;
}
