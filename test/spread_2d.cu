#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <algorithm>
#include <cuspreadinterp.h>
#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1, nf2;
	FLT upsampfac=2.0;
	int N1, N2, M;
	if (argc<5) {
		fprintf(stderr,"Usage: spread2d [method [maxsubprob [nupts_distr [N1 N2 [M [tol [kerevalmeth]]]]]]]\n");
		fprintf(stderr,"Details --\n");
		fprintf(stderr,"method 1: nupts driven\n");
		fprintf(stderr,"method 2: sub-problem\n");
		fprintf(stderr,"method 3: sub-problem with paul's idea\n");
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

	N1 = (int) nf1/upsampfac;
	N2 = (int) nf2/upsampfac;
	M = N1*N2;
	if(argc>6){
		sscanf(argv[6],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	FLT tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}

	int kerevalmeth=0;
	if(argc>8){
		sscanf(argv[8],"%d",&kerevalmeth);
	}

	int ier;

	int dim=2;
	int ns=std::ceil(-log10(tol/10.0));
	cufinufft_plan dplan;
	ier = cufinufft_default_opts(type1, dim, dplan.opts);
	if(ier != 0 ){
		cout<<"error: cufinufft_default_opts"<<endl;
		return 0;
	}
	ier = setup_spreader_for_nufft(dplan.spopts, tol, dplan.opts);
	dplan.opts.gpu_method=method;
	dplan.opts.upsampfac=upsampfac;
	dplan.opts.gpu_maxsubprobsize=maxsubprobsize;
	dplan.opts.gpu_kerevalmeth=kerevalmeth;
	dplan.spopts.pirange=0;

	cout<<scientific<<setprecision(3);


	FLT *x, *y;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fw,nf1*nf2*sizeof(CPX));

	switch(nupts_distribute){
		// Making data
		case 1: //uniform
			{
				for (int i = 0; i < M; i++) {
					x[i] = RESCALE(M_PI*randm11(), nf1, 1);// x in [-pi,pi)
					y[i] = RESCALE(M_PI*randm11(), nf2, 1);
					c[i].real(randm11());
					c[i].imag(randm11());
				}
			}
		case 2: // concentrate on a small region
			{
				for (int i = 0; i < M; i++) {
					x[i] = RESCALE(M_PI*rand01(), nf1, 1)/2.0 - 0.5;// x in [-pi,pi)
					y[i] = RESCALE(M_PI*rand01(), nf2, 1)/2.0 - 0.5;
					if(method == 6){
						x[i] = x[i] > nf1-0.5 ? x[i] - nf1 : x[i];
						y[i] = y[i] > nf2-0.5 ? y[i] - nf2 : y[i];// x in [-pi,pi)
					}
					c[i].real(randm11());
					c[i].imag(randm11());
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
	timer.restart();
	ier = cufinufft_spread2d(N1, N2, nf1, nf2, fw, M, x, y, c, &dplan);
	if(ier != 0 ){
		cout<<"error: cnufftspread2d"<<endl;
		return 0;
	}
	FLT t=timer.elapsedsec();
	printf("[Method %d] %ld NU pts to #%d U pts in %.3g s (%.3g NU pts/s)\n",
			dplan.opts.gpu_method,M,nf1*nf2,t,M/t);
#if 0
	cout<<"[result-input]"<<endl;
	for(int j=0; j<nf2; j++){
		if( j % dplan.opts.gpu_binsizey == 0)
			printf("\n");
		for (int i=0; i<nf1; i++){
			if( i % dplan.opts.gpu_binsizex == 0 && i!=0)
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
