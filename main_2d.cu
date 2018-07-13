#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "spread.h"
#include "utils.h"

using namespace std;

#define INFO
#define DEBUG
#define RESULT
#define TIME

int main(int argc, char* argv[])
{
  int N1, N2, M;
  if (argc<3) {
    fprintf(stderr,"Usage: spread2d [N1 N2 [M]]\n");
    return 1;
  }  
  double w;
  sscanf(argv[1],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
  sscanf(argv[2],"%lf",&w); N2 = (int)w;  // so can read 1e6 right!
  M = N1*N2;// let density always be 1
  if(argc>3){
    sscanf(argv[3],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
  }
  cout<<setprecision(5);
  FLT sigma = 2.0;
  int bin_size_x = 4;// for now, mod(nf1, bin_size_x) == 0
  int bin_size_y = 4;
  int nf1 = (int) sigma*N1;
  int nf2 = (int) sigma*N2;
  int ier;

  FLT *x, *y;
  CPX *c, *fwi, *fwo, *fwtrue;
  cudaMallocHost(&x, M*sizeof(CPX));
  cudaMallocHost(&y, M*sizeof(CPX));
  cudaMallocHost(&c, M*sizeof(CPX));
  cudaMallocHost(&fwi, nf1*nf2*sizeof(CPX));
  cudaMallocHost(&fwo, nf1*nf2*sizeof(CPX));
  cudaMallocHost(&fwtrue, nf1*nf2*sizeof(CPX));
  for (int i = 0; i < M; i++) {
    x[i] = RESCALE(M_PI*randm11(),nf1,1);// x in [-pi,pi)
    y[i] = RESCALE(M_PI*randm11(),nf2,1);
    c[i] = crandm11();
    //printf("(x,y)=(%f, %f)\n",x[i], y[i]);
  }
  CNTime timer;
  /*warm up gpu*/
  char *a;
  timer.restart();
  checkCudaErrors(cudaMalloc(&a,1));
#ifdef TIME
  cout<<"[time  ]"<< " (warm up) First cudamalloc call " << timer.elapsedsec() <<" s"<<endl<<endl;
#endif

#if 1
#ifdef INFO
  cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
#endif
  timer.restart();
  ier = cnufftspread2d_gpu_idriven(nf1, nf2, (FLT*) fwi, M, x, y, (FLT*) c);
  //ier = cnufftspread2d_gpu_odriven(nf1, nf2, fwi, M, x, y, c, 4, 4);
  FLT tidriven=timer.elapsedsec();
#ifdef TIME
  printf("[idriven] %ld NU pts to (%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",M,N1,N2,tidriven,M/tidriven);
#endif
#endif
  cout<<endl;


/* ------------------------------------------------------------------------------------------------------*/
#ifdef INFO
  cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
  cout<<"[info  ] Dividing the uniform grids to bin size["<<bin_size_x<<"x"<<bin_size_y<<"]"<<endl;
#endif
  timer.restart();
  ier = cnufftspread2d_gpu_odriven(nf1, nf2, fwo, M, x, y, c, bin_size_x, bin_size_y);
  FLT todriven=timer.elapsedsec();
#ifdef TIME
  printf("[odriven] %ld NU pts to (%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",M,N1,N2,todriven,M/todriven);
#endif
#if 0
  int ns=7;
  FLT es_c=4.0/(ns*ns);
  FLT es_beta=2.30*(FLT)ns;
  for(int j=0;j<nf2;j++){
    for(int i=0;i<nf1;i++){
      for(int k=0;k<M;k++){
        FLT xscaled=x[k];
        FLT yscaled=y[k];
        FLT disx = abs(xscaled-i) > nf1-ns/2.0 ? nf1-abs(xscaled-i) : abs(xscaled-i);
        FLT disy = abs(yscaled-j) > nf2-ns/2.0 ? nf2-abs(yscaled-j) : abs(yscaled-j);
        if( disx < ns/2.0 && disy < ns/2.0){
             //FLT kervalue1 = exp(es_beta * (sqrt(1.0 - es_c*disx*disx) - 1));
             //FLT kervalue2 = exp(es_beta * (sqrt(1.0 - es_c*disy*disy) - 1));
             FLT kervalue1 = 1;
             FLT kervalue2 = 1;
             fwtrue[i+j*nf1].real()+=kervalue1*kervalue2;
             fwtrue[i+j*nf1].imag()+=kervalue1*kervalue2;
        }
      }
    }
  }
  cout<<"[result-true]"<<endl;
  for(int j=0; j<nf2; j++){
    if( j % bin_size_y == 0)
        printf("\n");
    for (int i=0; i<nf1; i++){
      if( i % bin_size_x == 0 && i!=0)
        printf(" |");
      printf(" (%2.3g,%2.3g)",fwtrue[i+j*nf1].real(),fwtrue[i+j*nf1].imag() );
      //cout<<" "<<setw(8)<<fwo[i+j*nf1];
    }
    cout<<endl;
  }
  cout<<endl;
#endif
  cout<<endl;
  printf("[info  ] rel error between input driven, and out driven methods = %f\n", relerrtwonorm(nf1*nf2,fwi,fwo));

#ifdef RESULT
  cout<<"[resultdiff]"<<endl;
  for(int j=0; j<nf2; j++){
    for (int i=0; i<nf1; i++){
      if( norm(fwi[i+j*nf1]-fwo[i+j*nf1]) > 1e-8){
         cout<<norm(fwi[i+j*nf1]-fwo[i+j*nf1])<<" ";
         cout<<"(i,j)=("<<i<<","<<j<<"), "<<fwi[i+j*nf1] <<","<<fwo[i+j*nf1]<<endl;
      }
    }
  }
  cout<<endl;
#endif
#if 0
  cout<<"[result-input]"<<endl;
  for(int j=0; j<nf2; j++){
    if( j % bin_size_y == 0)
        printf("\n");
    for (int i=0; i<nf1; i++){
      if( i % bin_size_x == 0 && i!=0)
        printf(" |");
      printf(" (%2.3g,%2.3g)",fwi[i+j*nf1].real(),fwi[i+j*nf1].imag() );
      //cout<<" "<<setw(8)<<fwo[i+j*nf1];
    }
    cout<<endl;
  }
  cout<<endl;
#endif
#if 0
  cout<<"[result-output]"<<endl;
  for(int j=0; j<nf2; j++){
    if( j % bin_size_y == 0)
        printf("\n");
    for (int i=0; i<nf1; i++){
      if( i % bin_size_x == 0 && i!=0)
        printf(" |");
      printf(" (%2.3g,%2.3g)",fwo[i+j*nf1].real(),fwo[i+j*nf1].imag() );
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
