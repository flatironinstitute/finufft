#include <math.h>
#include "../src/utils.h"
#include "../src/finufft.h"
#include "../src/dirft.h"
#include <stdio.h>

// C++ stuff
#include <iostream>
#include <iomanip>
#include <vector>

// how big a problem to check direct DFT for...
#define BIGPROB 1e8

int main(int argc, char* argv[])
/* Test executable for finufft1d.

   Example: finufft1d_test 1000000 1000000 1e-12

   Barnett 1/22/17
*/
{
  BIGINT M = 1e6, N = 1e6;    // defaults: M = # srcs, N = # modes out
  double w, tol = 1e-6;          // default
  nufft_opts opts;
  opts.debug = 1;            // to see some timings
  opts.spread_debug = 0;     // see output from spreader
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) { sscanf(argv[1],"%lf",&w); N = (BIGINT)w; }
  if (argc>2) { sscanf(argv[2],"%lf",&w); M = (BIGINT)w; }
  if (argc>3) {
    sscanf(argv[3],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc==1 || argc>4) {
    fprintf(stderr,"Usage: finufft1d_test [Nmodes [Nsrc [tol]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);

  double *x = (double *)malloc(sizeof(double)*M);        // NU pts
  for (BIGINT j=0; j<M; ++j) x[j] = M_PI*randm11();
  //for (BIGINT j=0; j<M; ++j) x[j] = M_PI*(2*j/(double)M-1);  // test a grid
  dcomplex* c = (dcomplex*)malloc(sizeof(dcomplex)*M);   // strengths 
  dcomplex* F = (dcomplex*)malloc(sizeof(dcomplex)*N);   // mode ampls

  printf("test 1d type-1:\n"); // -------------- type 1
  for (BIGINT j=0; j<M; ++j) c[j] = crandm11();
  CNTime timer; timer.start();
  int ier = finufft1d1(M,x,(double*)c,isign,tol,N,(double*)F,opts);
  //for (int j=0;j<N;++j) cout<<F[j]<<endl;
  double t=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t%ld NU pts to %ld modes in %.3g s \t%.3g NU pts/s\n",M,N,t,M/t);

  BIGINT nt = N/2 - 7;      // check arbitrary choice of mode near the top
  dcomplex Ft = {0,0};
  for (BIGINT j=0; j<M; ++j)
    Ft += c[j] * exp(ima*((double)(isign*nt))*x[j]); // crude direct
  Ft /= M;
  printf("one mode: rel err in F[%ld] is %.3g\n",nt,abs(Ft-F[N/2+nt])/infnorm(N,F));
  if (M*N<=BIGPROB) {                  // also full direct eval
    dcomplex* Ft = (dcomplex*)malloc(sizeof(dcomplex)*N);
    dirft1d1(M,x,c,isign,N,Ft);
    printf("dirft1d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    free(Ft);
  }

  printf("test 1d type-2:\n"); // -------------- type 2
  for (BIGINT m=0; m<N; ++m) F[m] = crandm11();
  timer.restart();
  ier = finufft1d2(M,x,(double*)c,isign,tol,N,(double*)F,opts);
  //cout<<"c:\n"; for (int j=0;j<M;++j) cout<<c[j]<<endl;
  t=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t%ld modes to %ld NU pts in %.3g s \t%.3g NU pts/s\n",N,M,t,M/t);

  BIGINT jt = M/2;          // check arbitrary choice of one targ pt
  dcomplex ct = {0,0};
  BIGINT k0 = N/2;          // index shift in fk's = mag of most neg freq
  for (BIGINT m=-k0; m<=(N-1)/2; ++m)
    ct += F[k0+m] * exp(ima*((double)(isign*m))*x[jt]);   // crude direct
  printf("one targ: rel err in c[%ld] is %.3g\n",jt,abs(ct-c[jt])/infnorm(M,c));
  if (M*N<=BIGPROB) {                  // also full direct eval
    dcomplex* ct = (dcomplex*)malloc(sizeof(dcomplex)*M);
    dirft1d2(M,x,ct,isign,N,F);
    printf("dirft1d: rel l2-err of result c is %.3g\n",relerrtwonorm(M,ct,c));
    //cout<<"c/ct:\n"; for (int j=0;j<M;++j) cout<<c[j]/ct[j]<<endl;
    free(ct);
  }

  // --- todo: type 3



  free(x); free(c); free(F);
  return ier;
}
