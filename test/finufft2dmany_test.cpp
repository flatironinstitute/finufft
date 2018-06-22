#include "../src/finufft.h"
#include "../src/dirft.h"
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

// how big a problem to do full direct DFT check in 2D...
#define BIGPROB 1e8

// for omp rand filling
#define CHUNK 1000000

int main(int argc, char* argv[])
/* Test executable for finufft in 2d, all 3 types

   Usage: finufft2d_test [Nmodes1 Nmodes2 [Nsrc [tol [debug [spread_sort [upsampfac]]]]]]

   debug = 0: rel errors and overall timing, 1: timing breakdowns
           2: also spreading output

   Example: finufft2d_test 1000 1000 1000000 1e-12 1 2 2.0

   Barnett 2/1/17
*/
{
  BIGINT M = 1e6, N1 = 1000, N2 = 500;  // defaults: M = # srcs, N1,N2 = # modes
  int ndata = 400;
  double w, tol = 1e-6;          // default
  double upsampfac = 2.0;    // default
  nufft_opts opts; finufft_default_opts(opts);
  opts.debug = 0;            // 1 to see some timings
  // opts.fftw = FFTW_MEASURE;  // change from usual FFTW_ESTIMATE
  int isign = +1;             // choose which exponential sign to test
  if (argc>1) { sscanf(argv[1],"%lf",&w); ndata = (int)w; }
  if (argc>2) {
    sscanf(argv[2],"%lf",&w); N1 = (BIGINT)w;
    sscanf(argv[3],"%lf",&w); N2 = (BIGINT)w;
  }
  if (argc>4) { sscanf(argv[4],"%lf",&w); M = (BIGINT)w; }
  if (argc>5) {
    sscanf(argv[5],"%lf",&tol);
    if (tol<=0.0) { printf("tol must be positive!\n"); return 1; }
  }
  if (argc>6) sscanf(argv[6],"%d",&opts.debug);
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc>7) sscanf(argv[7],"%d",&opts.spread_sort);
  if (argc>8) sscanf(argv[8],"%d",&opts.many_seq);
  if (argc>9) sscanf(argv[9],"%lf",&upsampfac);
  opts.upsampfac=(FLT)upsampfac;

  if (argc==1 || argc==2 || argc>10) {
    fprintf(stderr,"Usage: finufft2d_test [ndata [N1 N2 [Nsrc [tol [debug [spread_sort [many_seq [upsampfac]]]]]]\n");
    return 1;
  }
  cout << scientific << setprecision(15);
  BIGINT N = N1*N2;

  FLT* x = (FLT*)malloc(sizeof(FLT)*M);  // NU pts x coords
  FLT* y = (FLT*)malloc(sizeof(FLT)*M);  // NU pts y coords
  CPX* c = (CPX*)malloc(sizeof(CPX)*M*ndata);   // strengths 
  CPX* F = (CPX*)malloc(sizeof(CPX)*N*ndata);   // mode ampls

  unsigned int se = 1;
  for (BIGINT j=0; j<M; ++j) {
    x[j] = M_PI*randm11r(&se);
    y[j] = M_PI*randm11r(&se);
  }

  for (BIGINT k = 0; k<ndata; ++k)
  {
    for (BIGINT j=0; j<M; ++j) {
      c[j+k*M] = crandm11r(&se);
    }
  }

  printf("test 2dmany type-1:\n"); // -------------- type 1
  CNTime timer; timer.start();
  int ier = finufft2d1many(ndata,M,x,y,c,isign,tol,N1,N2,F,opts);
  double ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t%d data: \t%ld NU pts to (%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",
	   ndata,(BIGINT)M,(BIGINT)N1,(BIGINT)N2,ti,ndata*M/ti);
#if 1
  // compare the result with finufft2d1
  CPX* cstart;
  CPX* F_finufft2d1 = (CPX*)malloc(sizeof(CPX)*N*ndata);
  double maxerror = 0.0;
  for (BIGINT k= 0; k<ndata; ++k)
  {
    cstart = c+k*M;
    ier = finufft2d1(M,x,y,cstart,isign,tol,N1,N2,F_finufft2d1,opts);
    maxerror = max(maxerror, relerrtwonorm(N,F_finufft2d1,F+k*N));
  }
  printf("max_data (  || F - F_finufft2d1 ||_2 / || F_finufft2d1 ||_2  ) =  %f\n",maxerror);
  free(F_finufft2d1);
#endif
  printf("test 2dmany type-2:\n"); // -------------- type 2

  for (BIGINT m=0; m<N*ndata; ++m) 
    F[m] = crandm11r(&se);
  timer.restart();
  ier = finufft2d2many(ndata,M,x,y,c,isign,tol,N1,N2,F,opts);
  ti=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
  } else
    printf("\t%d data: (%ld,%ld) modes to %ld NU pts in %.3g s \t%.3g NU pts/s\n",
           ndata,(BIGINT)N1,(BIGINT)N2,(BIGINT)M,ti,ndata*M/ti);
#if 1
  // compare the result with finufft2d1
  CPX* Fstart;
  CPX* c_finufft2d2 = (CPX*)malloc(sizeof(CPX)*M);
  maxerror = 0.0;
  for (BIGINT k= 0; k<ndata; ++k)
  {
    Fstart = F+k*N;
    ier = finufft2d2(M,x,y,c_finufft2d2,isign,tol,N1,N2,Fstart,opts);
    maxerror = max(maxerror, relerrtwonorm(M,c_finufft2d2,c+k*M));
  }
  printf("max_data ( || c - c_finufft2d1 ||_2 / || c_finufft2d1 ||_2 ) =  %f\n",maxerror);
  free(c_finufft2d2);
#endif
  free(x); free(y); free(c); free(F);
  return ier;
}
