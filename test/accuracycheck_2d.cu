#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "../src/spread.h"
#include "../src/finufft/utils.h"
#include "../src/finufft/cnufftspread.h"

using namespace std;

int main(int argc, char* argv[])
{
  int nf1, nf2;
  FLT sigma = 2.0;
  int N1, N2, M;
  if (argc<3) {
    fprintf(stderr,"Usage: spread2d [N1 N2 [M [tol[use_thrust]]]]\n");
    return 1;
  }  
  double w;
  sscanf(argv[1],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
  sscanf(argv[2],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!

  N1 = (int) nf1/sigma;
  N2 = (int) nf2/sigma;
  M = N1*N2;// let density always be 1
  if(argc>3){
    sscanf(argv[3],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
  }
  
  FLT tol=1e-6;
  if(argc>4){
    sscanf(argv[4],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
  }

  int use_thrust=0;
  if(argc>5){
    sscanf(argv[5],"%d",&use_thrust);
  }

  int ns=std::ceil(-log10(tol/10.0));
  spread_opts opts;
  opts.nspread=ns;
  opts.upsampfac=2.0;
  opts.ES_beta= 2.30 * (FLT)ns;
  opts.ES_c=4.0/(ns*ns);
  opts.ES_halfwidth=(FLT)ns/2;
  opts.use_thrust=use_thrust;

  cout<<setprecision(5);
  int ier;
  

  FLT *x, *y;
  CPX *c, *fwic, *fwfinufft;
  cudaMallocHost(&x, M*sizeof(CPX));
  cudaMallocHost(&y, M*sizeof(CPX));
  cudaMallocHost(&c, M*sizeof(CPX));
  cudaMallocHost(&fwic,      nf1*nf2*sizeof(CPX));
  cudaMallocHost(&fwfinufft, nf1*nf2*sizeof(CPX));
  for (int i = 0; i < M; i++) {
    x[i] = RESCALE(M_PI*randm11(), nf1, 1);// x in [-pi,pi)
    y[i] = RESCALE(M_PI*randm11(), nf2, 1);
    c[i].real() = randm11();
    c[i].imag() = randm11();
  }

  CNTime timer;
  /*warm up gpu*/
  char *a;
  timer.restart();
  checkCudaErrors(cudaMalloc(&a,1));
#ifdef TIME
  cout<<"[time  ]"<< " (warm up) First cudamalloc call " << timer.elapsedsec() <<" s"<<endl<<endl;
#endif

#ifdef INFO
  cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
#endif
/* ------------------------------------------------------------------------------------------------------*/
  timer.restart();
  opts.bin_size_x=32;
  opts.bin_size_y=32;
  ier = cnufftspread2d_gpu_hybrid(nf1, nf2, fwic, M, x, y, c, opts);
  FLT ticdriven=timer.elapsedsec();
  printf("[isorted] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
          M,N1,N2,nf1*nf2,ticdriven,M/ticdriven);
  cout<<endl;

  timer.start();
  opts.pirange = 0;
  opts.spread_direction=1;
  opts.flags=0;//ker always return 1
  opts.kerevalmeth=0;
  ier = cnufftspread(nf1,nf2,1,(FLT*) fwfinufft,M,x,y,NULL,(FLT*) c,opts);
  FLT t=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else
    printf("    %.3g NU pts in %.3g s \t%.3g pts/s \t%.3g spread pts/s\n",(double)M,t,M/t,pow(opts.nspread,2)*M/t);
  
  FLT err=relerrtwonorm(nf1*nf2,fwic,fwfinufft);
  printf("|| fwcuda - fwfinufft ||_2 / || fwcuda ||_2 =  %.3g\n", err);

  cudaFreeHost(x);
  cudaFreeHost(c);
  cudaFreeHost(fwic);
  cudaFreeHost(fwfinufft);
  return 0;
}
