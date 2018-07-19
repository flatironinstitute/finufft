#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "spread.h"
#include "utils.h"

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
  CPX *c, *fwi, *fwo, *fwtrue, *fwic, *fwh;
  cudaMallocHost(&x, M*sizeof(CPX));
  cudaMallocHost(&y, M*sizeof(CPX));
  cudaMallocHost(&c, M*sizeof(CPX));
  cudaMallocHost(&fwi, nf1*nf2*sizeof(CPX));
  cudaMallocHost(&fwic, nf1*nf2*sizeof(CPX));
  cudaMallocHost(&fwo, nf1*nf2*sizeof(CPX));
  cudaMallocHost(&fwh, nf1*nf2*sizeof(CPX));
  cudaMallocHost(&fwtrue, nf1*nf2*sizeof(CPX));
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
  timer.restart();
  ier = cnufftspread2d_gpu_idriven(nf1, nf2, fwi, M, x, y, c, opts);
  if(ier != 0 ){
    cout<<"error: cnufftspread2d_gpu_idriven"<<endl;
    return 0;
  }
  //ier = cnufftspread2d_gpu_odriven(nf1, nf2, fwi, M, x, y, c, 4, 4);
  FLT tidriven=timer.elapsedsec();
  printf("[idriven] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
         M,N1,N2,nf1*nf2,tidriven,M/tidriven);
  cout<<endl;

/* ------------------------------------------------------------------------------------------------------*/
  timer.restart();
  ier = cnufftspread2d_gpu_idriven_sorted(nf1, nf2, fwic, M, x, y, c, opts);
  FLT ticdriven=timer.elapsedsec();
  printf("[isorted] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
          M,N1,N2,nf1*nf2,ticdriven,M/ticdriven);
  cout<<endl;

/* ------------------------------------------------------------------------------------------------------*/
#if 0
  timer.restart();
  opts.bin_size_x=4;
  opts.bin_size_y=4;
  if(nf1 % opts.bin_size_x != 0 || nf2 % opts.bin_size_y !=0){
    cout << "error: mod(nf1,block_size_x) and mod(nf2,block_size_y) should be 0" << endl;
    return 0;
  }
#ifdef INFO
  cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
  cout<<"[info  ] Dividing the uniform grids to bin size["<<opts.bin_size_x<<"x"<<opts.bin_size_y<<"]"<<endl;
#endif
  ier = cnufftspread2d_gpu_odriven(nf1, nf2, fwo, M, x, y, c, opts);
  if(ier != 0 ){
    cout<<"error: cnufftspread2d_gpu_odriven"<<endl;
    return 0;
  }
  FLT todriven=timer.elapsedsec();
  printf("[odriven] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
         M,N1,N2,nf1*nf2,todriven,M/todriven);
  cout<<endl;
#endif

/*---------------------------------------------------------------------------------------------------------*/
#if 1
  timer.restart();
  opts.bin_size_x=32;
  opts.bin_size_y=32;
#ifdef INFO
  cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
  cout<<"[info  ] Dividing the uniform grids to bin size["<<opts.bin_size_x<<"x"<<opts.bin_size_y<<"]"<<endl;
#endif
  ier = cnufftspread2d_gpu_hybrid(nf1, nf2, fwh, M, x, y, c, opts);
  if(ier != 0 ){
    cout<<"error: cnufftspread2d_gpu_hybrid"<<endl;
    return 0;
  }
  FLT thybrid=timer.elapsedsec();
  printf("[hybrid ] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
         M,N1,N2,nf1*nf2,thybrid,M/thybrid);

#ifdef RESULT
  cout<<"[resultdiff]"<<endl;
  for(int j=0; j<nf2; j++){
    for (int i=0; i<nf1; i++){
      if( norm(fwic[i+j*nf1]-fwh[i+j*nf1]) > 1e-8){
         cout<<norm(fwic[i+j*nf1]-fwh[i+j*nf1])<<" ";
         cout<<"(i,j)=("<<i<<","<<j<<"), "<<fwic[i+j*nf1] <<","<<fwh[i+j*nf1]<<endl;
      }
    }
  }
  cout<<endl;
#endif
#endif
#if 0
  opts.bin_size_x=4;
  opts.bin_size_y=4;
  cout<<"[result-input]"<<endl;
  for(int j=0; j<nf2; j++){
    if( j % opts.bin_size_y == 0)
        printf("\n");
    for (int i=0; i<nf1; i++){
      if( i % opts.bin_size_x == 0 && i!=0)
        printf(" |");
      printf(" (%2.3g,%2.3g)",fwi[i+j*nf1].real(),fwi[i+j*nf1].imag() );
      //cout<<" "<<setw(8)<<fwo[i+j*nf1];
    }
    cout<<endl;
  }
  cout<<endl;
#endif
#if 0
  opts.bin_size_x=32;
  opts.bin_size_y=32;
  cout<<"[result-output]"<<endl;
  for(int j=0; j<nf2; j++){
    if( j % opts.bin_size_y == 0)
        printf("\n");
    for (int i=0; i<nf1; i++){
      if( i % opts.bin_size_x == 0 && i!=0)
        printf(" |");
      printf(" (%2.3g,%2.3g)",fwh[i+j*nf1].real(),fwh[i+j*nf1].imag() );
      //cout<<" "<<setw(8)<<fwo[i+j*nf1];
    }
    cout<<endl;
  }
  cout<<endl;
#endif
  cudaFreeHost(x);
  cudaFreeHost(c);
  cudaFreeHost(fwi);
  cudaFreeHost(fwo);
  cudaFreeHost(fwtrue);
  return 0;
}
