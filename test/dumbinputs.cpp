#include "../src/finufft.h"
#include "../src/dirft.h"
#include <complex>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int main(int argc, char* argv[])
/* calling the FINUFFT library from C++ using all manner of crazy inputs that
   might cause errors. All should be caught gracefully.
   Barnett 3/14/17.

   Compile with:
   g++ -fopenmp dumbinputs.cpp ../lib/libfinufft.a -o dumbinputs  -lfftw3 -lfftw3_omp -lm
   or if you have built a single-core version:
   g++ dumbinputs.cpp ../lib/libfinufft.a -o dumbinputs -lfftw3 -lm

   Usage: ./dumbinputs
*/
{
  int M = 100;            // number of nonuniform points
  int N = 10;             // # modes, keep small, also output NU pts in type 3
  FLT acc = 1e-6;         // desired accuracy
  nufft_opts opts; finufft_default_opts(&opts);   // recommended

  int NN = N*N*N;         // modes F alloc size since we'll go to 3d
  // generate some "random" nonuniform points (x) and complex strengths (c):
  FLT *x = (FLT *)malloc(sizeof(FLT)*M);
  CPX* c = (CPX*)malloc(sizeof(CPX)*M);
  for (int j=0; j<M; ++j) {
    x[j] = PI*cos((FLT)j);                           // deterministic
    c[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j);
  }
  // allocate output array F for Fourier modes, fix some type-3 coords...
  CPX* F = (CPX*)malloc(sizeof(CPX)*NN);
  FLT *s = (FLT*)malloc(sizeof(FLT)*N);
  for (int k=0; k<N; ++k) s[k] = 10 * cos(1.2*k);   // normal-sized coords
  FLT *shuge = (FLT*)malloc(sizeof(FLT)*N);
  FLT huge = 1e11;                                  // no smaller than MAX_NF
  for (int k=0; k<N; ++k) shuge[k] = huge * s[k];   // some huge coords

  // alloc exact output array
  CPX* Fe = (CPX*)malloc(sizeof(CPX)*NN);
 
  // some useful debug printing...
  //for (int k=0;k<N;++k) printf("F[%d] = %g+%gi\n",k,real(F[k]),imag(F[k]));
  //for (int j=0;j<M;++j) printf("c[%d] = %g+%gi\n",j,real(c[j]),imag(c[j]));
  //printf("%.3g %3g\n",twonorm(N,F),twonorm(M,c));
  opts.debug = 0;   // set to 1,2, to debug segfaults
  opts.spread_debug = 0;

  printf("1D dumb cases ----------------\n");
  int ier = finufft1d1(M,x,c,+1,0,N,F,opts);
  printf("1d1 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft1d1(M,x,c,+1,acc,0,F,opts);
  printf("1d1 N=0:\tier=%d\n",ier);
  ier = finufft1d1(0,x,c,+1,acc,N,F,opts);
  printf("1d1 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  for (int k=0; k<NN; ++k) F[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // set F for t2
  ier = finufft1d2(M,x,c,+1,0,N,F,opts);
  printf("1d2 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft1d2(M,x,c,+1,acc,0,F,opts);
  printf("1d2 N=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft1d2(0,x,c,+1,acc,N,F,opts);
  printf("1d2 M=0:\tier=%d\n",ier);
  for (int j=0; j<M; ++j) c[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset c for t3
  ier = finufft1d3(M,x,c,+1,0,N,s,F,opts);
  printf("1d3 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft1d3(M,x,c,+1,acc,0,s,F,opts);
  printf("1d3 nk=0:\tier=%d\n",ier);
  ier = finufft1d3(0,x,c,+1,acc,N,s,F,opts);
  printf("1d3 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  ier = finufft1d3(1,x,c,+1,acc,N,s,F,opts);   // XK prod formally 0
  dirft1d3(1,x,c,+1,N,s,Fe); for (int k=0; k<N; ++k) F[k] -= Fe[k]; // acc chk
  printf("1d3 M=1:\tier=%d\tnrm(err)=%.4f\n",ier,twonorm(N,F));  // to 1e-4 abs
  ier = finufft1d3(M,x,c,+1,acc,1,s,F,opts);   // "
  dirft1d3(M,x,c,+1,1,s,Fe);
  printf("1d3 N=1:\tier=%d\terr=%.4f\n",ier,abs(F[0]-Fe[0]));
  ier = finufft1d3(1,x,c,+1,acc,1,s,F,opts);   // "
  dirft1d3(1,x,c,+1,1,s,Fe);
  printf("1d3 M=N=1:\tier=%d\terr=%.4f\n",ier,abs(F[0]-Fe[0]));
  ier = finufft1d3(M,x,c,+1,acc,N,shuge,F,opts);
  printf("1d3 XK prod too big:\tier=%d (should complain)\n",ier);

  printf("2D dumb cases ----------------\n"); // (uses y=x, and t=s in type 3)
  ier = finufft2d1(M,x,x,c,+1,0,N,N,F,opts);
  printf("2d1 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft2d1(M,x,x,c,+1,acc,0,0,F,opts);
  printf("2d1 Ns=Nt=0:\tier=%d\n",ier);
  ier = finufft2d1(M,x,x,c,+1,acc,0,N,F,opts);
  printf("2d1 Ns=0,Nt>0:\tier=%d\n",ier);
  ier = finufft2d1(M,x,x,c,+1,acc,N,0,F,opts);
  printf("2d1 Ns>0,Ns=0:\tier=%d\n",ier);
  ier = finufft2d1(0,x,x,c,+1,acc,N,N,F,opts);
  printf("2d1 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  for (int k=0; k<NN; ++k) F[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // set F for t2
  ier = finufft2d2(M,x,x,c,+1,0,N,N,F,opts);
  printf("2d2 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft2d2(M,x,x,c,+1,acc,0,0,F,opts);
  printf("2d2 Ns=Nt=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft2d2(M,x,x,c,+1,acc,0,N,F,opts);
  printf("2d2 Ns=0,Nt>0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft2d2(M,x,x,c,+1,acc,N,0,F,opts);
  printf("2d2 Ns>0,Nt=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft2d2(0,x,x,c,+1,acc,N,N,F,opts);
  printf("2d2 M=0:\tier=%d\n",ier);
  for (int j=0; j<M; ++j) c[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset c for t3
  ier = finufft2d3(M,x,x,c,+1,0,N,s,s,F,opts);
  printf("2d3 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft2d3(M,x,x,c,+1,acc,0,s,s,F,opts);
  printf("2d3 nk=0:\tier=%d\n",ier);
  ier = finufft2d3(0,x,x,c,+1,acc,N,s,s,F,opts);
  printf("2d3 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  ier = finufft2d3(1,x,x,c,+1,acc,N,s,s,F,opts);   // XK prod formally 0
  printf("2d3 M=1:\tier=%d\tnrm(F)=%.3g\n",ier,twonorm(N,F));
  for (int k=0; k<N; ++k) shuge[k] = sqrt(huge)*s[k];     // less huge coords
  ier = finufft2d3(M,x,x,c,+1,acc,N,shuge,shuge,F,opts);
  printf("2d3 XK prod too big:\tier=%d (should complain)\n",ier);

  int ndata = 10;
  CPX* cm = (CPX*)malloc(sizeof(CPX)*M*ndata);
  CPX* Fm = (CPX*)malloc(sizeof(CPX)*NN*ndata);
  for (int j=0; j<M*ndata; ++j) cm[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // set cm for 2d1many
  ier = finufft2d1many(0,M,x,x,cm,+1,0,N,N,Fm,opts);
  printf("2d1many ndata=0:\tier=%d (should complain)\n",ier);
  ier = finufft2d1many(ndata,M,x,x,cm,+1,0,N,N,Fm,opts);
  printf("2d1many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = finufft2d1many(ndata,M,x,x,cm,+1,acc,0,0,Fm,opts);
  printf("2d1many Ns=Nt=0:\tier=%d\n",ier);
  ier = finufft2d1many(ndata,M,x,x,cm,+1,acc,0,N,Fm,opts);
  printf("2d1many Ns=0,Nt>0:\tier=%d\n",ier);
  ier = finufft2d1many(ndata,M,x,x,cm,+1,acc,N,0,Fm,opts);
  printf("2d1many Ns>0,Ns=0:\tier=%d\n",ier);
  ier = finufft2d1many(ndata,0,x,x,cm,+1,acc,N,N,Fm,opts);
  printf("2d1many M=0:\t\tier=%d\tnrm(Fm)=%.3g (should vanish)\n",ier,twonorm(N*ndata,Fm));
  for (int k=0; k<NN*ndata; ++k) Fm[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // set F for t2
  ier = finufft2d2many(0,M,x,x,cm,+1,0,N,N,Fm,opts);
  printf("2d2many ndata=0:\tier=%d (should complain)\n",ier);
  ier = finufft2d2many(ndata,M,x,x,cm,+1,0,N,N,Fm,opts);
  printf("2d2many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = finufft2d2many(ndata,M,x,x,cm,+1,acc,0,0,Fm,opts);
  printf("2d2many Ns=Nt=0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = finufft2d2many(ndata,M,x,x,cm,+1,acc,0,N,Fm,opts);
  printf("2d2many Ns=0,Nt>0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = finufft2d2many(ndata,M,x,x,cm,+1,acc,N,0,Fm,opts);
  printf("2d2many Ns>0,Nt=0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = finufft2d2many(ndata,0,x,x,cm,+1,acc,N,N,Fm,opts);
  printf("2d2many M=0:\t\tier=%d\n",ier);


  printf("3D dumb cases ----------------\n");    // z=y=x, and u=t=s in type 3
  ier = finufft3d1(M,x,x,x,c,+1,0,N,N,N,F,opts);
  printf("3d1 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft3d1(M,x,x,x,c,+1,acc,0,0,0,F,opts);
  printf("3d1 Ns=Nt=Nu=0:\tier=%d\n",ier);
  ier = finufft3d1(M,x,x,x,c,+1,acc,0,N,N,F,opts);
  printf("3d1 Ns,Nt>0,Nu=0:\tier=%d\n",ier);
  ier = finufft3d1(M,x,x,x,c,+1,acc,N,0,N,F,opts);
  printf("3d1 Ns>0,Nt=0,Nu>0:\tier=%d\n",ier);
  ier = finufft3d1(M,x,x,x,c,+1,acc,N,N,0,F,opts);
  printf("3d1 Ns,Nt>0,Nu=0:\tier=%d\n",ier);
  ier = finufft3d1(0,x,x,x,c,+1,acc,N,N,N,F,opts);
  printf("3d1 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  for (int k=0; k<NN; ++k) F[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // set F for t2
  ier = finufft3d2(M,x,x,x,c,+1,0,N,N,N,F,opts);
  printf("3d2 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft3d2(M,x,x,x,c,+1,acc,0,0,0,F,opts);
  printf("3d2 Ns=Nt=Nu=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft3d2(M,x,x,x,c,+1,acc,0,N,N,F,opts);
  printf("3d2 Ns=0,Nt,Nu>0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft3d2(M,x,x,x,c,+1,acc,N,0,N,F,opts);
  printf("3d2 Ns>0,Nt=0,Nu>0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft3d2(M,x,x,x,c,+1,acc,N,N,0,F,opts);
  printf("3d2 Ns,Nt>0,Nu=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = finufft3d2(0,x,x,x,c,+1,acc,N,N,N,F,opts);
  printf("3d2 M=0:\tier=%d\n",ier);
  for (int j=0; j<M; ++j) c[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset c for t3
  ier = finufft3d3(M,x,x,x,c,+1,0,N,s,s,s,F,opts);
  printf("3d3 tol=0:\tier=%d (should complain)\n",ier);
  ier = finufft3d3(M,x,x,x,c,+1,acc,0,s,s,s,F,opts);
  printf("3d3 nk=0:\tier=%d\n",ier);
  ier = finufft3d3(0,x,x,x,c,+1,acc,N,s,s,s,F,opts);
  printf("3d3 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  ier = finufft3d3(1,x,x,x,c,+1,acc,N,s,s,s,F,opts);   // XK prod formally 0
  printf("3d3 M=1:\tier=%d\tnrm(F)=%.3g\n",ier,twonorm(N,F));
  for (int k=0; k<N; ++k) shuge[k] = pow(huge,1./3)*s[k];  // less huge coords
  ier = finufft3d3(M,x,x,x,c,+1,acc,N,shuge,shuge,shuge,F,opts);
  printf("3d3 XK prod too big:\tier=%d (should complain)\n",ier);

  free(x); free(c); free(F); free(s); free(shuge); free(cm); free(Fm);
  printf("freed.\n");
  return 0;
}
