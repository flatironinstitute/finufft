#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <algorithm>
#include "../src/cufinufft.h"
#include "../src/spreadinterp.h"
#include "../finufft/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1, nf2;
	FLT sigma = 2.0;
	int N1, N2, M;
	if (argc<5) {
		fprintf(stderr,"Usage: spread2d [method [maxsubprob [nupts_distr [N1 N2 [rep [tol [kerevalmeth]]]]]]]\n");
		fprintf(stderr,"Details --\n");
		fprintf(stderr,"method 1: input driven without sorting\n");
		fprintf(stderr,"method 2: input driven with sorting\n");
		fprintf(stderr,"method 4: hybrid\n");
		fprintf(stderr,"method 5: subprob\n");
		return 1;
	}  
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	int maxsubprobsize;
	sscanf(argv[2],"%d",&maxsubprobsize);
	int nupts_distribute;
	sscanf(argv[3],"%d",&nupts_distribute);
	sscanf(argv[4],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[5],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!

	N1 = (int) nf1/sigma;
	N2 = (int) nf2/sigma;
	int rep = 10;
	if(argc>6){
		//sscanf(argv[6],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
		sscanf(argv[6],"%d",&rep);
		//if(M == 0) M=N1*N2*4*rep;
	}
	M = N1*N2*4*rep;// let density always be 1
	M = nf1*nf2*rep;// let density always be 1

	FLT tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}

	int kerevalmeth=0;
	if(argc>8){
		sscanf(argv[8],"%d",&kerevalmeth);
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


	FLT *x, *y;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fw,nf1*nf2*sizeof(CPX));

	opts.pirange=0;
	opts.kerevalmeth=kerevalmeth;
	if(method == 6)
		opts.gpu_maxsubprobsize=maxsubprobsize;
	if(method == 5)
		opts.gpu_maxsubprobsize=maxsubprobsize;
	switch(nupts_distribute){
		// Making data
		case 1: //uniform
			{
				for (int j=0; j<nf2; j++) {
					for (int i=0; i<nf1; i++){
						for (int k=0; k<rep; k++){
							if(k+i*rep+j*nf1*rep < M){
								x[k+i*rep+j*nf1*rep] = i;
								y[k+i*rep+j*nf1*rep] = j;
							}
						}
					}
				}
#if 0 
				srand(unsigned(1)); 
				random_shuffle (&x[0], &x[M-1]);
				srand(unsigned(1)); 
				random_shuffle (&y[0], &y[M-1]);
#endif
				for (int i = 0; i < M; i++) {
					c[i].real() = randm11();
					c[i].imag() = randm11();
				}
			}
			break;

		case 2:
			{
				for (int k=0; k<rep; k++){
					for (int j=0; j<nf2; j++) {
						for (int i=0; i<nf1; i++){
							if(i+j*nf1+k*nf1*nf2< M){
								x[i+j*nf1+k*nf1*nf2] = i;
								y[i+j*nf1+k*nf1*nf2] = j;
							}
						}
					}
				}
				for (int i = 0; i < M; i++) {
					c[i].real() = randm11();
					c[i].imag() = randm11();
				}
			}
			break;
		
		case 3:
			{
				for (int j=0; j<nf2; j++) {
					for (int i=0; i<nf1; i++){
						for (int k=0; k<rep; k++){
							if(k+i*rep+j*nf1*rep < M){
								x[k+i*rep+j*nf1*rep] = i;
								y[k+i*rep+j*nf1*rep] = j;
							}
						}
					}
				}
				srand(unsigned(1)); 
				random_shuffle (&x[0], &x[M-1]);
				srand(unsigned(1)); 
				random_shuffle (&y[0], &y[M-1]);
				for (int i = 0; i < M; i++) {
					c[i].real() = randm11();
					c[i].imag() = randm11();
				}
			}
			break;
		case 4: // concentrate on a small region
			{
				for (int i = 0; i < M; i++) {
					x[i] = RESCALE(M_PI*rand01(), nf1, 1)/2.0 - 0.5;// x in [-pi,pi)
					y[i] = RESCALE(M_PI*rand01(), nf2, 1)/2.0 - 0.5;
					if(method == 6){
						x[i] = x[i] > nf1-0.5 ? x[i] - nf1 : x[i];
						y[i] = y[i] > nf2-0.5 ? y[i] - nf2 : y[i];// x in [-pi,pi)
					}
					c[i].real() = randm11();
					c[i].imag() = randm11();
				}
			}
			break;
		case 5:
			{
				for (int i = 0; i < M; i++) {
					x[i] = RESCALE(M_PI*randm11(), nf1, 1);// x in [-pi,pi)
					y[i] = RESCALE(M_PI*randm11(), nf2, 1);
					if(method == 6){
						x[i] = x[i] > nf1-0.5 ? x[i] - nf1 : x[i];
						y[i] = y[i] > nf2-0.5 ? y[i] - nf2 : y[i];// x in [-pi,pi)
					}
					c[i].real() = randm11();
					c[i].imag() = randm11();
				}
			}
			break;
		case 6:
			{
				for(int i=0; i<M; i++) {
					x[i] = 1;// x in [-pi,pi)
					y[i] = 1;
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
	cout<<"[time  ]"<< " (warm up) First cudamalloc call " << timer.elapsedsec() 
		<<" s"<<endl<<endl;
#endif

#ifdef INFO
	cout<<"[info  ] Spreading "<<M<<" pts to ["<<nf1<<"x"<<nf2<<"] uniform grids"
		<<endl;
#endif
	if(opts.gpu_method == 2)
	{
		opts.gpu_binsizex=16;
		opts.gpu_binsizey=16;
	}

	if(opts.gpu_method == 4 || opts.gpu_method==5)
	{
		opts.gpu_binsizex=32;
		opts.gpu_binsizey=32;
	}

	if(opts.gpu_method == 6)
	{
		opts.gpu_binsizex=32;
		opts.gpu_binsizey=32;
	}
	timer.restart();
	ier = cufinufft_spread2d(N1, N2, nf1, nf2, fw, M, x, y, c, opts, &dplan);
	if(ier != 0 ){
		cout<<"error: cnufftspread2d"<<endl;
		return 0;
	}
	FLT t=timer.elapsedsec();
	printf("[Method %d] %ld NU pts to #%d U pts in %.3g s (\t%.3g NU pts/s)\n",
			opts.gpu_method,M,nf1*nf2,t,M/t);
#if 0
	switch(method)
	{
		case 4:
			opts.gpu_binsizex=32;
			opts.gpu_binsizey=32;
		case 5:
			opts.gpu_binsizex=16;
			opts.gpu_binsizey=16;
		default:
			opts.gpu_binsizex=nf1;
			opts.gpu_binsizey=nf2;		
	}
	cout<<"[result-input]"<<endl;
	for(int j=0; j<nf2; j++){
		if( j % opts.gpu_binsizey == 0)
			printf("\n");
		for (int i=0; i<nf1; i++){
			if( i % opts.gpu_binsizex == 0 && i!=0)
				printf(" |");
			printf(" (%2.3g,%2.3g)",fw[i+j*nf1].real(),fw[i+j*nf1].imag() );
		}
		cout<<endl;
	}
	cout<<endl;
#endif

	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fw);
	return 0;
}
		
