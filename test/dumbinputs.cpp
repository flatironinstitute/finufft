#include <test_defs.h>
// this enforces recompilation, responding to SINGLE...
#include "directft/dirft1d.cpp"
#include "directft/dirft2d.cpp"
#include "directft/dirft3d.cpp"
using namespace std;

int main(int argc, char* argv[])
/* calling the FINUFFT library from C++ using all manner of crazy inputs that
   might cause errors. Simple and "many" interfaces mostly, with 2 guru cases
   at the end (need more). All bad inputs should be caught gracefully.
   (It also checks accuracy for 1D type 3, for some reason - could be killed.)
   Barnett 3/14/17, updated Andrea Malleo, summer 2019.
   Libin Lu switch to use ptr-to-opts interfaces, Feb 2020.
   guru: makeplan followed by immediate destroy. Barnett 5/26/20.
   Either precision with dual-prec lib funcs 7/3/20.
   Added a chkbnds case to 1d1, 4/9/21.

   Compile with (better to go up a directory and use: make test/dumbinputs) :
   g++ -std=c++14 -fopenmp dumbinputs.cpp -I../include ../lib/libfinufft.so -o dumbinputs  -lfftw3 -lfftw3_omp -lm

   or if you have built a single-core version:
   g++ -std=c++14 dumbinputs.cpp -I../include ../lib/libfinufft.so -o dumbinputs -lfftw3 -lm

   Usage: ./dumbinputs
   
   Output file will say "(should complain)" if that ier should be >0.

   Also compare (diff) against test/results/dumbinputs.refout
*/
{
  int M = 100;            // number of nonuniform points
  int N = 10;             // # modes, keep small, also output NU pts in type 3
  FLT acc = 1e-6;         // desired accuracy
  nufft_opts opts; FINUFFT_DEFAULT_OPTS(&opts);

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
  int ier = FINUFFT1D1(M,x,c,+1,0,N,F,&opts);
  printf("1d1 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D1(M,x,c,+1,acc,0,F,&opts);
  printf("1d1 N=0:\tier=%d\n",ier);
  ier = FINUFFT1D1(0,x,c,+1,acc,N,F,&opts);
  printf("1d1 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  FLT xsave = x[0];
  x[0] = 3*PI*(1 + 2*EPSILON);   // works in either prec, just outside valid
  ier = FINUFFT1D1(M,x,c,+1,acc,N,F,&opts);
  printf("1d1 x>3pi:\tier=%d (should complain)\n",ier);
  x[0] = INFINITY;
  ier = FINUFFT1D1(M,x,c,+1,acc,N,F,&opts);
  printf("1d1 x=Inf:\tier=%d (should complain)\n",ier);
  x[0] = NAN;
  ier = FINUFFT1D1(M,x,c,+1,acc,N,F,&opts);
  printf("1d1 x=NaN:\tier=%d (should complain)\n",ier);
  x[0] = xsave;
  
  for (int k=0; k<NN; ++k) F[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // set F for t2
  ier = FINUFFT1D2(M,x,c,+1,0,N,F,&opts);
  printf("1d2 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D2(M,x,c,+1,acc,0,F,&opts);
  printf("1d2 N=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = FINUFFT1D2(0,x,c,+1,acc,N,F,&opts);
  printf("1d2 M=0:\tier=%d\n",ier);

  for (int j=0; j<M; ++j) c[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset c for t3
  ier = FINUFFT1D3(M,x,c,+1,0,N,s,F,&opts);
  printf("1d3 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D3(M,x,c,+1,acc,0,s,F,&opts);
  printf("1d3 nk=0:\tier=%d\n",ier);
  ier = FINUFFT1D3(0,x,c,+1,acc,N,s,F,&opts);
  printf("1d3 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  ier = FINUFFT1D3(1,x,c,+1,acc,N,s,F,&opts);   // XK prod formally 0
  dirft1d3(1,x,c,+1,N,s,Fe); for (int k=0; k<N; ++k) F[k] -= Fe[k]; // acc chk
  printf("1d3 M=1:\tier=%d\tnrm(err)=%.4f\n",ier,twonorm(N,F));  // to 1e-4 abs
  ier = FINUFFT1D3(M,x,c,+1,acc,1,s,F,&opts);   // "
  dirft1d3(M,x,c,+1,1,s,Fe);
  printf("1d3 N=1:\tier=%d\terr=%.4f\n",ier,abs(F[0]-Fe[0]));
  ier = FINUFFT1D3(1,x,c,+1,acc,1,s,F,&opts);   // "
  dirft1d3(1,x,c,+1,1,s,Fe);
  printf("1d3 M=N=1:\tier=%d\terr=%.4f\n",ier,abs(F[0]-Fe[0]));
  ier = FINUFFT1D3(M,x,c,+1,acc,N,shuge,F,&opts);
  printf("1d3 XK prod too big:\tier=%d (should complain)\n",ier);

  int ndata = 10;                 // how many multiple vectors to test it on
  CPX* cm = (CPX*)malloc(sizeof(CPX)*M*ndata);
  CPX* Fm = (CPX*)malloc(sizeof(CPX)*NN*ndata);     // the biggest array
  for (int j=0; j<M*ndata; ++j) cm[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // set cm for 1d1many
  ier = FINUFFT1D1MANY(0,M,x,cm,+1,0,N,Fm,&opts);
  printf("1d1many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D1MANY(ndata,M,x,cm,+1,0,N,Fm,&opts);
  printf("1d1many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D1MANY(ndata,M,x,cm,+1,acc,0,Fm,&opts);
  printf("1d1many Ns=0:\tier=%d\n",ier);
  ier = FINUFFT1D1MANY(ndata,0,x,cm,+1,acc,N,Fm,&opts);
  printf("1d1many M=0:\t\tier=%d\tnrm(Fm)=%.3g (should vanish)\n",ier,twonorm(N*ndata,Fm));
  
  for (int k=0; k<NN*ndata; ++k) Fm[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // set Fm for 1d2many
  ier = FINUFFT1D2MANY(0,M,x,cm,+1,0,N,Fm,&opts);
  printf("1d2many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D2MANY(ndata,M,x,cm,+1,0,N,Fm,&opts);
  printf("1d2many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D2MANY(ndata,M,x,cm,+1,acc,0,Fm,&opts);
  printf("1d2many Ns=0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",ier,twonorm(M*ndata,cm));
  ier = FINUFFT1D2MANY(ndata,0,x,cm,+1,acc,N,Fm,&opts);
  printf("1d2many M=0:\t\tier=%d\n",ier);

  for (int j=0; j<M*ndata; ++j) cm[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset cm for 1d3many
  ier = FINUFFT1D3MANY(0, M,x,c,+1,acc,N,s,Fm,&opts);
  printf("1d3many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D3MANY(ndata, M,x,c,+1,0,N,s,Fm,&opts);
  printf("1d3many tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT1D3MANY(ndata, M,x,c,+1,acc,0,s,Fm,&opts);
  printf("1d3many nk=0:\tier=%d\n",ier);
  ier = FINUFFT1D3MANY(ndata, 0,x,c,+1,acc,N,s,Fm,&opts);
  printf("1d3many M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,Fm));
  ier = FINUFFT1D3MANY(ndata, 1,x,c,+1,acc,N,s,Fm,&opts);   // XK prod formally 0
  dirft1d3(1,x,c,+1,N,s,Fe); for (int k=0; k<N; ++k) Fm[k] -= Fe[k]; // acc chk
  printf("1d3many M=1:\tier=%d\tnrm(err)=%.4f\n",ier,twonorm(N,Fm));  // to 1e-4 abs; check just first trial
  ier = FINUFFT1D3MANY(ndata,M,x,c,+1,acc,1,s,Fm,&opts);   // "
  dirft1d3(M,x,c,+1,1,s,Fe);
  printf("1d3many N=1:\tier=%d\terr=%.4f\n",ier,abs(Fm[0]-Fe[0]));
  ier = FINUFFT1D3MANY(ndata,1,x,c,+1,acc,1,s,Fm,&opts);   // "
  dirft1d3(1,x,c,+1,1,s,Fe);
  printf("1d3many M=N=1:\tier=%d\terr=%.4f\n",ier,abs(Fm[0]-Fe[0]));
  ier = FINUFFT1D3MANY(ndata,M,x,c,+1,acc,N,shuge,F,&opts);
  printf("1d3many XK prod too big:\tier=%d (should complain)\n",ier);

  printf("2D dumb cases ----------------\n"); // (uses y=x, and t=s in type 3)
  ier = FINUFFT2D1(M,x,x,c,+1,0,N,N,F,&opts);
  printf("2d1 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D1(M,x,x,c,+1,acc,0,0,F,&opts);
  printf("2d1 Ns=Nt=0:\tier=%d\n",ier);
  ier = FINUFFT2D1(M,x,x,c,+1,acc,0,N,F,&opts);
  printf("2d1 Ns=0,Nt>0:\tier=%d\n",ier);
  ier = FINUFFT2D1(M,x,x,c,+1,acc,N,0,F,&opts);
  printf("2d1 Ns>0,Ns=0:\tier=%d\n",ier);
  ier = FINUFFT2D1(0,x,x,c,+1,acc,N,N,F,&opts);
  printf("2d1 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));

  for (int k=0; k<NN; ++k) F[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // set F for t2
  ier = FINUFFT2D2(M,x,x,c,+1,0,N,N,F,&opts);
  printf("2d2 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D2(M,x,x,c,+1,acc,0,0,F,&opts);
  printf("2d2 Ns=Nt=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = FINUFFT2D2(M,x,x,c,+1,acc,0,N,F,&opts);
  printf("2d2 Ns=0,Nt>0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = FINUFFT2D2(M,x,x,c,+1,acc,N,0,F,&opts);
  printf("2d2 Ns>0,Nt=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = FINUFFT2D2(0,x,x,c,+1,acc,N,N,F,&opts);
  printf("2d2 M=0:\tier=%d\n",ier);

  for (int j=0; j<M; ++j) c[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset c for t3
  ier = FINUFFT2D3(M,x,x,c,+1,0,N,s,s,F,&opts);
  printf("2d3 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D3(M,x,x,c,+1,acc,0,s,s,F,&opts);
  printf("2d3 nk=0:\tier=%d\n",ier);
  ier = FINUFFT2D3(0,x,x,c,+1,acc,N,s,s,F,&opts);
  printf("2d3 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  ier = FINUFFT2D3(1,x,x,c,+1,acc,N,s,s,F,&opts);   // XK prod formally 0
  printf("2d3 M=1:\tier=%d\tnrm(F)=%.3g\n",ier,twonorm(N,F));
  for (int k=0; k<N; ++k) shuge[k] = sqrt(huge)*s[k];     // less huge coords
  ier = FINUFFT2D3(M,x,x,c,+1,acc,N,shuge,shuge,F,&opts);
  printf("2d3 XK prod too big:\tier=%d (should complain)\n",ier);

  for (int j=0; j<M*ndata; ++j) cm[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset cm for 2d1many
  ier = FINUFFT2D1MANY(0,M,x,x,cm,+1,0,N,N,Fm,&opts);
  printf("2d1many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D1MANY(ndata,M,x,x,cm,+1,0,N,N,Fm,&opts);
  printf("2d1many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D1MANY(ndata,M,x,x,cm,+1,acc,0,0,Fm,&opts);
  printf("2d1many Ns=Nt=0:\tier=%d\n",ier);
  ier = FINUFFT2D1MANY(ndata,M,x,x,cm,+1,acc,0,N,Fm,&opts);
  printf("2d1many Ns=0,Nt>0:\tier=%d\n",ier);
  ier = FINUFFT2D1MANY(ndata,M,x,x,cm,+1,acc,N,0,Fm,&opts);
  printf("2d1many Ns>0,Ns=0:\tier=%d\n",ier);
  ier = FINUFFT2D1MANY(ndata,0,x,x,cm,+1,acc,N,N,Fm,&opts);
  printf("2d1many M=0:\t\tier=%d\tnrm(Fm)=%.3g (should vanish)\n",ier,twonorm(N*ndata,Fm));

  for (int k=0; k<NN*ndata; ++k) Fm[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // reset Fm for t2
  ier = FINUFFT2D2MANY(0,M,x,x,cm,+1,0,N,N,Fm,&opts);
  printf("2d2many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D2MANY(ndata,M,x,x,cm,+1,0,N,N,Fm,&opts);
  printf("2d2many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D2MANY(ndata,M,x,x,cm,+1,acc,0,0,Fm,&opts);
  printf("2d2many Ns=Nt=0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = FINUFFT2D2MANY(ndata,M,x,x,cm,+1,acc,0,N,Fm,&opts);
  printf("2d2many Ns=0,Nt>0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = FINUFFT2D2MANY(ndata,M,x,x,cm,+1,acc,N,0,Fm,&opts);
  printf("2d2many Ns>0,Nt=0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = FINUFFT2D2MANY(ndata,0,x,x,cm,+1,acc,N,N,Fm,&opts);
  printf("2d2many M=0:\t\tier=%d\n",ier);

  ier = FINUFFT2D3MANY(0,M,x,x,cm,+1,0,N,s,s,Fm,&opts);
  printf("2d3many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D3MANY(ndata,M,x,x,cm,+1,0,N,s,s,Fm,&opts);
  printf("2d3many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = FINUFFT2D3MANY(ndata,M,x,x,cm,+1,acc,0,s,s,Fm,&opts);
  printf("2d3many nk=0:\tier=%d\n", ier);
  ier = FINUFFT2D3MANY(ndata,0,x,x,cm,+1,acc,N,s,s,Fm,&opts);
  printf("2d3many M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,Fm));
  ier = FINUFFT2D3MANY(ndata,1,x,x,c,+1,acc,N,s,s,F,&opts);   // XK prod formally 0
  printf("2d3 M=1:\tier=%d\tnrm(F)=%.3g\n",ier,twonorm(N,Fm));
  ier = FINUFFT2D3MANY(ndata,M,x,x,c,+1,acc,N,shuge,shuge,Fm,&opts);
  printf("2d3many XK prod too big:\tier=%d (should complain)\n",ier);


  printf("3D dumb cases ----------------\n");    // z=y=x, and u=t=s in type 3
  ier = FINUFFT3D1(M,x,x,x,c,+1,0,N,N,N,F,&opts);
  printf("3d1 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D1(M,x,x,x,c,+1,acc,0,0,0,F,&opts);
  printf("3d1 Ns=Nt=Nu=0:\tier=%d\n",ier);
  ier = FINUFFT3D1(M,x,x,x,c,+1,acc,0,N,N,F,&opts);
  printf("3d1 Ns,Nt>0,Nu=0:\tier=%d\n",ier);
  ier = FINUFFT3D1(M,x,x,x,c,+1,acc,N,0,N,F,&opts);
  printf("3d1 Ns>0,Nt=0,Nu>0:\tier=%d\n",ier);
  ier = FINUFFT3D1(M,x,x,x,c,+1,acc,N,N,0,F,&opts);
  printf("3d1 Ns,Nt>0,Nu=0:\tier=%d\n",ier);
  ier = FINUFFT3D1(0,x,x,x,c,+1,acc,N,N,N,F,&opts);
  printf("3d1 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));

  for (int k=0; k<NN; ++k) F[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // set F for t2
  ier = FINUFFT3D2(M,x,x,x,c,+1,0,N,N,N,F,&opts);
  printf("3d2 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D2(M,x,x,x,c,+1,acc,0,0,0,F,&opts);
  printf("3d2 Ns=Nt=Nu=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = FINUFFT3D2(M,x,x,x,c,+1,acc,0,N,N,F,&opts);
  printf("3d2 Ns=0,Nt,Nu>0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = FINUFFT3D2(M,x,x,x,c,+1,acc,N,0,N,F,&opts);
  printf("3d2 Ns>0,Nt=0,Nu>0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = FINUFFT3D2(M,x,x,x,c,+1,acc,N,N,0,F,&opts);
  printf("3d2 Ns,Nt>0,Nu=0:\tier=%d\tnrm(c)=%.3g (should vanish)\n",ier,twonorm(M,c));
  ier = FINUFFT3D2(0,x,x,x,c,+1,acc,N,N,N,F,&opts);
  printf("3d2 M=0:\tier=%d\n",ier);

  for (int j=0; j<M; ++j) c[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset c for t3
  ier = FINUFFT3D3(M,x,x,x,c,+1,0,N,s,s,s,F,&opts);
  printf("3d3 tol=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D3(M,x,x,x,c,+1,acc,0,s,s,s,F,&opts);
  printf("3d3 nk=0:\tier=%d\n",ier);
  ier = FINUFFT3D3(0,x,x,x,c,+1,acc,N,s,s,s,F,&opts);
  printf("3d3 M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,F));
  ier = FINUFFT3D3(1,x,x,x,c,+1,acc,N,s,s,s,F,&opts);   // XK prod formally 0
  printf("3d3 M=1:\tier=%d\tnrm(F)=%.3g\n",ier,twonorm(N,F));
  for (int k=0; k<N; ++k) shuge[k] = pow(huge,1./3)*s[k];  // less huge coords
  ier = FINUFFT3D3(M,x,x,x,c,+1,acc,N,shuge,shuge,shuge,F,&opts);
  printf("3d3 XK prod too big:\tier=%d (should complain)\n",ier);

  for (int j=0; j<M*ndata; ++j) cm[j] = sin((FLT)1.3*j) + IMA*cos((FLT)0.9*j); // reset cm for 3d1many
  ier = FINUFFT3D1MANY(0,M,x,x,x,cm,+1,0,N,N,N,Fm,&opts);
  printf("3d1many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D1MANY(ndata,M,x,x,x,cm,+1,0,N,N,N,Fm,&opts);
  printf("3d1many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D1MANY(ndata,M,x,x,x,cm,+1,acc,0,0,0,Fm,&opts);
  printf("3d1many Ns=Nt=Nu=0:\tier=%d\n",ier);
  ier = FINUFFT3D1MANY(ndata,M,x,x,x,cm,+1,acc,0,N,N,Fm,&opts);
  printf("3d1many Ns=0,Nt>0,Nu>0:\tier=%d\n",ier);
  ier = FINUFFT3D1MANY(ndata,M,x,x,x,cm,+1,acc,N,0,N,Fm,&opts);
  printf("3d1many Ns>0,Ns=0,Nu>0:\tier=%d\n",ier);
  ier = FINUFFT3D1MANY(ndata,M,x,x,x,cm,+1,acc,N,N,0,Fm,&opts);
  printf("3d1many Ns>0,Ns>0,Nu=0,:\tier=%d\n",ier);
  ier = FINUFFT3D1MANY(ndata,0,x,x,x,cm,+1,acc,N,N,N,Fm,&opts);
  printf("3d1many M=0:\t\tier=%d\tnrm(Fm)=%.3g (should vanish)\n",ier,twonorm(N*ndata,Fm));

  for (int k=0; k<NN*ndata; ++k) Fm[k] = sin((FLT)0.7*k) + IMA*cos((FLT)0.3*k);  // reset Fm for t2
  ier = FINUFFT3D2MANY(0,M,x,x,x,cm,+1,0,N,N,N,Fm,&opts);
  printf("3d2many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D2MANY(ndata,M,x,x,x,cm,+1,0,N,N,N,Fm,&opts);
  printf("3d2many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D2MANY(ndata,M,x,x,x,cm,+1,acc,0,0,0,Fm,&opts);
  printf("3d2many Ns=Nt=Nu=0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = FINUFFT3D2MANY(ndata,M,x,x,x,cm,+1,acc,0,N,N,Fm,&opts);
  printf("3d2many Ns=0,Nt>0,Nu>0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = FINUFFT3D2MANY(ndata,M,x,x,x,cm,+1,acc,N,0,N,Fm,&opts);
  printf("3d2many Ns>0,Nt=0,Nu>0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = FINUFFT3D2MANY(ndata,M,x,x,x,cm,+1,acc,N,N,0,Fm,&opts);
  printf("3d2many Ns>0,Nt>0,Nu=0:\tier=%d\tnrm(cm)=%.3g (should vanish)\n",
  	      ier,twonorm(M*ndata,cm));
  ier = FINUFFT3D2MANY(ndata,0,x,x,x,cm,+1,acc,N,N,N,Fm,&opts);
  printf("3d2many M=0:\t\tier=%d\n",ier);

  ier = FINUFFT3D3MANY(0,M,x,x,x,cm,+1,0,N,s,s,s,Fm,&opts);
  printf("3d3many ndata=0:\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D3MANY(ndata,M,x,x,x,cm,+1,0,N,s,s,s,Fm,&opts);
  printf("3d3many tol=0:\t\tier=%d (should complain)\n",ier);
  ier = FINUFFT3D3MANY(ndata,M,x,x,x,cm,+1,acc,0,s,s,s,Fm,&opts);
  printf("3d3many nk=0:\tier=%d\n", ier);
  ier = FINUFFT3D3MANY(ndata,0,x,x,x,cm,+1,acc,N,s,s,s,Fm,&opts);
  printf("3d3many M=0:\tier=%d\tnrm(F)=%.3g (should vanish)\n",ier,twonorm(N,Fm));
  ier = FINUFFT3D3MANY(ndata,1,x,x,x,c,+1,acc,N,s,s,s,F,&opts);   // XK prod formally 0
  printf("3d3 M=1:\tier=%d\tnrm(F)=%.3g\n",ier,twonorm(N,Fm));
  ier = FINUFFT3D3MANY(ndata,M,x,x,x,c,+1,acc,N,shuge,shuge,shuge,Fm,&opts);
  printf("3d3many XK prod too big:\tier=%d (should complain)\n",ier);
  
  free(x); free(c); free(F); free(s); free(shuge); free(cm); free(Fm);
  printf("freed.\n");
  
  // some dumb tests for guru interface to induce free() crash in destroy...
  FINUFFT_PLAN plan;
  BIGINT Ns[1] = {0};      // since dim=1, don't have to make length 3
  FINUFFT_MAKEPLAN(1, 1, Ns, +1, 1, acc, &plan, NULL);  // type 1, now kill it
  FINUFFT_DESTROY(plan);
  FINUFFT_MAKEPLAN(3, 1, Ns, +1, 1, acc, &plan, NULL);  // type 3, now kill it
  FINUFFT_DESTROY(plan);
  // *** todo: more extensive bad inputs and error catching in guru...
  
  return 0;
}
