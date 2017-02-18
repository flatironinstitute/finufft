#include "../src/finufft.h"
#include "../src/dirft.h"
#include <math.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <iomanip>

// how big a problem to do full direct DFT check...
#define BIGPROB 1e8

int main(int argc, char* argv[])
/* Test executable for finufft3d.

   Example: finufft3d_test 100 200 50 1e6 1e-12

   Barnett 2/2/17
*/
{
  BIGINT M = 1e6, N1 = 100, N2 = 200, N3 = 50;  // defaults: M = # srcs, N1,N2,N3 = # modes
  double w, tol = 1e-6;          // default
  nufft_opts opts;
  opts.debug = 1;            // to see some timings
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) {
    sscanf(argv[1],"%lf",&w); N1 = (BIGINT)w;
    sscanf(argv[2],"%lf",&w); N2 = (BIGINT)w;
    sscanf(argv[3],"%lf",&w); N3 = (BIGINT)w;
  }
  if (argc>4) { sscanf(argv[4],"%lf",&w); M = (BIGINT)w; }
  if (argc>5) {
    sscanf(argv[5],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>6) sscanf(argv[6],"%d",&opts.debug);  // can be 0,1 or 2
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc==1 || argc==2 || argc==3 || argc>7) {
    fprintf(stderr,"Usage: finufft3d_test [N1 N2 N3 [Nsrc [tol [debug]]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);
  BIGINT N = N1*N2*N3;

  double *x = (double *)malloc(sizeof(double)*M);        // NU pts x coords
  double *y = (double *)malloc(sizeof(double)*M);        // NU pts y coords
  double *z = (double *)malloc(sizeof(double)*M);        // NU pts z coords
  for (BIGINT j=0; j<M; ++j) {
    x[j] = M_PI*randm11();
    y[j] = M_PI*randm11();
    z[j] = M_PI*randm11();
  }
  dcomplex* c = (dcomplex*)malloc(sizeof(dcomplex)*M);   // strengths 
  dcomplex* F = (dcomplex*)malloc(sizeof(dcomplex)*N);   // mode ampls

  printf("test 3d type-1:\n"); // -------------- type 1
  for (BIGINT j=0; j<M; ++j) c[j] = crandm11();
  CNTime timer; timer.start();
  int ier = finufft3d1(M,x,y,z,c,isign,tol,N1,N2,N3,F,opts);
  double ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("     %ld NU pts to (%ld,%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",
	   M,N1,N2,N3,ti,M/ti);

  BIGINT nt1 = N1/2 - 7, nt2 = N2/2 - 5, nt3 = N3/2 - 8;  // choose mode to check
  dcomplex Ft = {0,0}, J = ima*(double)isign;
  for (BIGINT j=0; j<M; ++j)
    Ft += c[j] * exp(J*(nt1*x[j]+nt2*y[j]+nt3*z[j]));   // crude direct
  Ft /= M;
  // index in complex F as 1d array...
  BIGINT it = N1/2+nt1 + N1*(N2/2+nt2) + N1*N2*(N3/2+nt3);
  printf("one mode: rel err in F[%ld,%ld,%ld] is %.3g\n",nt1,nt2,nt3,
	 abs(Ft-F[it])/infnorm(N,F));
  if (M*N<=BIGPROB) {                   // also check vs full direct eval
    dcomplex* Ft = (dcomplex*)malloc(sizeof(dcomplex)*N);
    dirft3d1(M,x,y,z,c,isign,N1,N2,N3,Ft);
    printf("dirft3d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    free(Ft);
  }

  printf("test 3d type-2:\n"); // -------------- type 2
  for (BIGINT m=0; m<N; ++m) F[m] = crandm11();
  timer.restart();
  ier = finufft3d2(M,x,y,z,c,isign,tol,N1,N2,N3,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("     (%ld,%ld,%ld) modes to %ld NU pts in %.3g s \t%.3g NU pts/s\n",
	   N1,N2,N3,M,ti,M/ti);

  BIGINT jt = M/2;          // check arbitrary choice of one targ pt
  dcomplex ct = {0,0};
  BIGINT m=0;
  for (BIGINT m3=-(N3/2); m3<=(N3-1)/2; ++m3)   // loop in F order
    for (BIGINT m2=-(N2/2); m2<=(N2-1)/2; ++m2)
      for (BIGINT m1=-(N1/2); m1<=(N1-1)/2; ++m1)
	ct += F[m++] * exp(J*(m1*x[jt] + m2*y[jt] + m3*z[jt]));   // direct
  printf("one targ: rel err in c[%ld] is %.3g\n",jt,abs(ct-c[jt])/infnorm(M,c));
  if (M*N<=BIGPROB) {                  // also full direct eval
    dcomplex* ct = (dcomplex*)malloc(sizeof(dcomplex)*M);
    dirft3d2(M,x,y,z,ct,isign,N1,N2,N3,F);
    printf("dirft3d: rel l2-err of result c is %.3g\n",relerrtwonorm(M,ct,c));
    free(ct);
  }

  printf("test 3d type-3:\n"); // -------------- type 3
  // reuse the strengths c, interpret N as number of targs:
  for (BIGINT j=0; j<M; ++j) {
    x[j] = 2.0 + M_PI*randm11();      // new x_j srcs, offset from origin
    y[j] = -3.0 + M_PI*randm11();     // " y_j
    z[j] = 1.0 + M_PI*randm11();     // " z_j
  }
  double* s = (double*)malloc(sizeof(double)*N);    // targ freqs (1-cmpt)
  double* t = (double*)malloc(sizeof(double)*N);    // targ freqs (2-cmpt)
  double* u = (double*)malloc(sizeof(double)*N);    // targ freqs (3-cmpt)
  double S1 = (double)N1/2;                   // choose freq range sim to type 1
  double S2 = (double)N2/2;
  double S3 = (double)N3/2;
  for (BIGINT k=0; k<N; ++k) {
    s[k] = S1*(1.7 + randm11());    //S*(1.7 + k/(double)N); // offset the freqs
    t[k] = S2*(-0.5 + randm11());
    u[k] = S3*(0.9 + randm11());
  }
  timer.restart();
  ier = finufft3d3(M,x,y,z,c,isign,tol,N,s,t,u,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t%ld NU to %ld NU in %.3g s   %.3g srcs/s, %.3g targs/s\n",M,N,ti,M/ti,N/ti);

  BIGINT kt = N/2;          // check arbitrary choice of one targ pt
  Ft = {0,0};
  for (BIGINT j=0;j<M;++j)
    Ft += c[j] * exp(ima*(double)isign*(s[kt]*x[j] + t[kt]*y[j] + u[kt]*z[j]));
  printf("one targ: rel err in F[%ld] is %.3g\n",kt,abs(Ft-F[kt])/infnorm(N,F));
  if (M*N<=BIGPROB) {                  // also full direct eval
    dcomplex* Ft = (dcomplex*)malloc(sizeof(dcomplex)*N);
    dirft3d3(M,x,y,z,c,isign,N,s,t,u,Ft);       // writes to F
    printf("dirft3d: rel l2-err of result F is %.3g\n",relerrtwonorm(N,Ft,F));
    //cout<<"s t u, F, Ft, F/Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<" "<<t[k]<<" "<<u[k]<<", "<<F[k]<<",\t"<<Ft[k]<<",\t"<<F[k]/Ft[k]<<endl;
    free(Ft);
  }

  free(x); free(y); free(z); free(c); free(F);
  return ier;
}
