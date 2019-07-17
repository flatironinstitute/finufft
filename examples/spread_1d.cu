#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "../src/cufinufft.h"
#include "../src/spreadinterp.h"
#include "../finufft/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1;
	FLT sigma = 2.0;
	int N1, M;
	if (argc<4) {
		fprintf(stderr,"Usage: spread1d [method [nupts_distr [N1 [M [tol]]]]]\n");
		fprintf(stderr,"Details --\n");
		fprintf(stderr,"method 1: input driven without sorting\n");
		fprintf(stderr,"method 5: subprob\n");
		return 1;
	}
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	int nupts_distribute;
	sscanf(argv[2],"%d",&nupts_distribute);
	sscanf(argv[3],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!

	N1 = (int) nf1/sigma;
	M = N1;// let density always be 1
	if(argc>4){
		sscanf(argv[4],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
		if(M == 0) M=N1;
	}

	FLT tol=1e-6;
	if(argc>5){
		sscanf(argv[5],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}

	int ier;

	int ns=std::ceil(-log10(tol/10.0));
	cufinufft_opts opts;
	cufinufft_plan dplan;
	FLT upsampfac=2.0;

	ier = cufinufft_default_opts(opts,tol,upsampfac);
  if(ier != 0 ){
    cout<<"error: cufinufft_default_opts"<<endl;
    return 0;
  }
	opts.gpu_method=method;
	cout<<scientific<<setprecision(3);


	FLT *x;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fw,nf1*sizeof(CPX));

	opts.pirange=0;
  switch(nupts_distribute){
    // Making data
    case 1: //uniform
    {
      for (int i = 0; i < M; i++) {
        x[i] = RESCALE(M_PI*randm11(), nf1, 1);// x in [-pi,pi)
        c[i].real() = randm11();
        c[i].imag() = randm11();
      }
    }
    break;
    case 2: // concentrate on a small region
    {
      for (int i = 0; i < M; i++) {
        x[i] = RESCALE(M_PI*rand01()/(nf1*2/32), nf1, 1);// x in [-pi,pi)
        c[i].real() = randm11();
        c[i].imag() = randm11();
      }
    }
    break;
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

	if(opts.gpu_method==5)
	{
		opts.gpu_binsizex=32;

	}

	timer.restart();
	ier = cufinufft_spread1d(N1, nf1, fw, M, x, c, opts, &dplan);
	if(ier != 0 ){
		cout<<"error: cnufftspread1d"<<endl;
		return 0;
	}
	FLT t=timer.elapsedsec();
	printf("[Method %d] %ld NU pts to #%d U pts in %.3g s (\t%.3g NU pts/s)\n",
		opts.gpu_method,M,nf1,t,M/t);
#ifdef RESULT
	switch(method)
	{
		case 5:
			opts.gpu_binsizex=32;
		default:
			opts.gpu_binsizex=nf1;
	}
	cout<<"[result-input]"<<endl;
	for (int i=0; i<nf1; i++){
		if( i % opts.gpu_binsizex == 0 && i!=0)
			printf(" |");
		printf(" (%2.3g,%2.3g)",fw[i].real(),fw[i].imag() );
	}
	cout<<endl;
#endif

	cudaFreeHost(x);
	cudaFreeHost(c);
	cudaFreeHost(fw);
	return 0;
}
