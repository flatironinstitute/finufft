#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "spread.h"
#include "utils.h"

using namespace std;

#define INFO
//#define DEBUG
//#define RESULT
#define TIME
#define OUTDRIVEN 0

int main(int argc, char* argv[])
{
  cout<<setprecision(3)<<endl;
  int N1 = 128, N2 = 128;
  int M = N1*N2;
  FLT sigma = 2.0;
  int bin_size_x = 16;
  int bin_size_y = 16;
  int nf1 = (int) sigma*N1;
  int nf2 = (int) sigma*N2;

  FLT *x, *y;
  complex<FLT> *c, *fw;
  x  = (FLT*) malloc(M*sizeof(FLT));
  y  = (FLT*) malloc(M*sizeof(FLT));
  c  = (complex<FLT>*) malloc(M*sizeof(complex<FLT>));
  fw = (complex<FLT>*) malloc(nf1*nf2*sizeof(complex<FLT>));

  for (int i = 0; i < M; i++) {
    x[i] = M_PI*randm11();// x in [-pi,pi)
    y[i] = M_PI*randm11();
    c[i] = crandm11();
  }
#ifdef INFO
  cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"<<endl;
  cout<<"[info  ] Dividing the uniform grids to bin size["<<bin_size_x<<"x"<<bin_size_y<<"]"<<endl;
#endif
  CNTime timer;
  /*warm up gpu*/
  char *a;
  timer.restart();
  checkCudaErrors(cudaMalloc(&a,1));
#ifdef TIME
  cout<<"[time  ]"<< " (warm up) First cudamalloc call " << timer.elapsedsec() <<" s"<<endl;
#endif

  int ier;
  timer.restart();
  ier = cnufftspread2d_gpu_idriven(nf1, nf2, (FLT*) fw, M, x, y, (FLT*) c);
  FLT tidriven=timer.elapsedsec();
  timer.restart();
  ier = cnufftspread2d_gpu_idriven(nf1, nf2, (FLT*) fw, M, x, y, (FLT*) c);
  FLT todriven=timer.elapsedsec();

#ifdef TIME
  printf("[info  ] [odriven] %ld NU pts to (%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",M,N1,N2,ti,M/todriven);
  printf("[info  ] [idriven] %ld NU pts to (%ld,%ld) modes in %.3g s \t%.3g NU pts/s\n",M,N1,N2,ti,M/tidriven);
#endif
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
  free(x);
  free(c);
  free(fw);
  return 0;
}
