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
//#define RESULT
#define TIME

int main(int argc, char* argv[])
{
  int N1, N2, M;
  if (argc!=3) {
    fprintf(stderr,"Usage: spread2d [N1 N2]\n");
    return 1;
  }  
  double w;
  sscanf(argv[1],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
  sscanf(argv[2],"%lf",&w); N2 = (int)w;  // so can read 1e6 right!
  //M = N1*N2;// let density always be 1
  M = N1*N2;
  cout<<setprecision(3);
  FLT sigma = 2.0;
  int bin_size_x = 4;
  int bin_size_y = 4;
  int nf1 = (int) sigma*N1;
  int nf2 = (int) sigma*N2;

  FLT *x, *y;
  CPX *c, *fwi, *fwo;
  cudaMallocHost(&x, M*sizeof(CPX));
  cudaMallocHost(&y, M*sizeof(CPX));
  cudaMallocHost(&c, M*sizeof(CPX));
  cudaMallocHost(&fwi, nf1*nf2*sizeof(CPX));
  cudaMallocHost(&fwo, nf1*nf2*sizeof(CPX));

  for (int i = 0; i < M; i++) {
    x[i] = M_PI*randm11();// x in [-pi,pi)
    y[i] = M_PI*randm11();
    c[i] = crandm11();
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
  int ier;
  timer.restart();
  ier = cnufftspread2d_gpu_idriven(nf1, nf2, (FLT*) fwi, M, x, y, (FLT*) c);
  FLT tidriven=timer.elapsedsec();
#ifdef TIME
  printf("[idriven] %ld NU pts to (%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",M,N1,N2,tidriven,M/tidriven);
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
  for(int j=0;j<nf2;j++){
    for(int i=0;i<nf1;i++){
      for(int k=0;k<M;k++){
        FLT xscaled=RESCALE(x[k],nf1,1);
        FLT yscaled=RESCALE(y[k],nf2,1);
        FLT disx = abs(xscaled-i) > nf1-ns/2.0 ? nf1-abs(xscaled-i) : abs(xscaled-i);
        FLT disy = abs(yscaled-j) > nf2-ns/2.0 ? nf2-abs(yscaled-j) : abs(yscaled-j);
        if( disx < ns/2.0 && disy < ns/2.0){
             fwtrue[i+j*nf1].real()++;
             fwtrue[i+j*nf1].imag()++;
             //FLT kervalue = evaluate_kernel(sqrt(disx*disx+disy*disy), es_c, es_beta);
             //tr += cshared[2*j]*kervalue;
             //ti += cshared[2*j+1]*kervalue;
        }
      }
    }
    cout<<j<<endl;
  }
  printf("%f\n", relerrtwonorm(nf1,fwi,fwtrue));
#endif
  cout<<endl;
  printf("[info  ] rel error between input driven, and out driven methods = %f\n", relerrtwonorm(nf1,fwi,fwo));

#ifdef RESULT
  cout<<"[result]"<<endl;
  for(int j=0; j<nf2; j++){
    if( j % bin_size_y == 0)
        cout<<endl;
    for (int i=0; i<nf1; i++){
      if( i % bin_size_x == 0 && i!=0)
        cout<< " |";
      cout<<" "<<setw(8)<<fw[i+j*nf1];
    }
    cout<<endl;
  }
  cout<<endl;
#endif
  cudaFreeHost(x);
  cudaFreeHost(c);
  cudaFreeHost(fwi);
  cudaFreeHost(fwo);
  return 0;
}
