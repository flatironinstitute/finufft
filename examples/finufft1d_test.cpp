#include "../src/finufft.h"
#include "../src/dirft.h"
#include <math.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <iomanip>

// how big a problem to check direct DFT for...
#define BIGPROB 1e8

int main(int argc, char* argv[])
/* Test executable for finufft1d, all 3 types.

   Usage: finufft1d_test [Nmodes [Nsrc [tol [debug]]]]

   Example: finufft1d_test 1000000 1000000 1e-12

   Barnett 1/22/17 - 2/9/17
*/
{
  BIGINT M = 1e6, N = 1e6;   // defaults: M = # srcs, N = # modes out
  double w, tol = 1e-6;      // default
  nufft_opts opts;
  opts.debug = 1;            // to see some timings
  int isign = +1;            // choose which exponential sign to test
  if (argc>1) { sscanf(argv[1],"%lf",&w); N = (BIGINT)w; }
  if (argc>2) { sscanf(argv[2],"%lf",&w); M = (BIGINT)w; }
  if (argc>3) {
    sscanf(argv[3],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>4) sscanf(argv[4],"%d",&opts.debug);
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc==1 || argc>5) {
    fprintf(stderr,"Usage: finufft1d_test [Nmodes [Nsrc [tol [debug]]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);

  double *x = (double *)malloc(sizeof(double)*M);        // NU pts
  for (BIGINT j=0; j<M; ++j) x[j] = M_PI*randm11();   // fills [-pi,pi)
  //for (BIGINT j=0; j<M; ++j) x[j] = 0.999 * M_PI*randm11();  // avoid ends
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
  BIGINT m=0, k0 = N/2;          // index shift in fk's = mag of most neg freq
  for (BIGINT m1=-k0; m1<=(N-1)/2; ++m1)
    ct += F[m++] * exp(ima*((double)(isign*m1))*x[jt]);   // crude direct
  printf("one targ: rel err in c[%ld] is %.3g\n",jt,abs(ct-c[jt])/infnorm(M,c));
  if (M*N<=BIGPROB) {                  // also full direct eval
    dcomplex* ct = (dcomplex*)malloc(sizeof(dcomplex)*M);
    dirft1d2(M,x,ct,isign,N,F);
    printf("dirft1d: rel l2-err of result c is %.3g\n",relerrtwonorm(M,ct,c));
    //cout<<"c/ct:\n"; for (int j=0;j<M;++j) cout<<c[j]/ct[j]<<endl;
    free(ct);
  }

  printf("test 1d type-3:\n"); // -------------- type 3
  // reuse the strengths c, interpret N as number of targs:
  for (BIGINT j=0; j<M; ++j) x[j] = 2.0 + M_PI*randm11();  // new x_j srcs
  double* s = (double*)malloc(sizeof(double)*N);    // targ freqs
  double S = (double)N/2;                   // choose freq range sim to type 1
  for (BIGINT k=0; k<N; ++k) s[k] = S*(1.7 + randm11()); //S*(1.7 + k/(double)N); // offset
  timer.restart();
  ier = finufft1d3(M,x,(double*)c,isign,tol,N,s,(double*)F,opts);
  t=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t%ld NU to %ld NU in %.3g s   %.3g srcs/s, %.3g targs/s\n",M,N,t,M/t,N/t);

  BIGINT kt = N/2;          // check arbitrary choice of one targ pt
  Ft = {0,0};
  for (BIGINT j=0;j<M;++j)
    Ft += c[j] * exp(ima*(double)isign*s[kt]*x[j]);
  printf("one targ: rel err in F[%ld] is %.3g\n",kt,abs(Ft-F[kt])/infnorm(N,F));
  if (M*N<=BIGPROB) {                  // also full direct eval
    dcomplex* Ft = (dcomplex*)malloc(sizeof(dcomplex)*N);
    dirft1d3(M,x,c,isign,N,s,Ft);       // writes to F
    printf("dirft1d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    //cout<<"s, F, Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<" "<<F[k]<<"\t"<<Ft[k]<<"\t"<<F[k]/Ft[k]<<endl;
    free(Ft);
  }

  free(x); free(c); free(F);
  return ier;
}
