#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "../src/spread.h"
#include "../src/finufft/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1, nf2;
	FLT sigma = 2.0;
	int N1, N2, M;
	if (argc<4) {
		fprintf(stderr,"Usage: spread2d [method [N1 N2 [M [tol [use_thrust]]]]]\n");
		fprintf(stderr,"Details --\n");
		fprintf(stderr,"method 1: input driven without sorting\n");
		fprintf(stderr,"method 2: input driven with sorting\n");
		fprintf(stderr,"method 3: output driven\n");
		fprintf(stderr,"method t: hybrid\n");
		return 1;
	}  
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	sscanf(argv[2],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[3],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!

	N1 = (int) nf1/sigma;
	N2 = (int) nf2/sigma;
	M = N1*N2;// let density always be 1
	if(argc>4){
		sscanf(argv[4],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	FLT tol=1e-6;
	if(argc>5){
		sscanf(argv[5],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}

	int use_thrust=0;
	if(argc>6){
		sscanf(argv[6],"%d",&use_thrust);
	}

	int ns=std::ceil(-log10(tol/10.0));
	spread_opts opts;
	opts.nspread=ns;
	opts.upsampfac=2.0;
	opts.ES_beta=2.30*(FLT)ns;
	opts.ES_c=4.0/(ns*ns);
	opts.ES_halfwidth=(FLT)ns/2;
	opts.use_thrust=use_thrust;

	cout<<scientific<<setprecision(16);
	int ier;


	FLT *x, *y;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(CPX));
	cudaMallocHost(&y, M*sizeof(CPX));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fw,nf1*nf2*sizeof(CPX));

	// Making data
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
	switch(method)
	{
		case 1:
		{
			timer.restart();
			ier = cnufftspread2d_gpu_idriven(nf1, nf2, fw, M, x, y, c, opts);
			if(ier != 0 ){
				cout<<"error: cnufftspread2d_gpu_idriven"<<endl;
				return 0;
			}
			FLT tidriven=timer.elapsedsec();
			printf("[idriven] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
					M,N1,N2,nf1*nf2,tidriven,M/tidriven);
		}
		break;
		case 2:
		{
			timer.restart();
			ier = cnufftspread2d_gpu_idriven_sorted(nf1, nf2, fw, M, x, y, c, opts);
			FLT ticdriven=timer.elapsedsec();
			printf("[isorted] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
					M,N1,N2,nf1*nf2,ticdriven,M/ticdriven);
		}
		break;
		case 3:
		{
			timer.restart();
			opts.bin_size_x=4;
			opts.bin_size_y=4;
			if(nf1 % opts.bin_size_x != 0 || nf2 % opts.bin_size_y !=0){
				cout << "error: mod(nf1,block_size_x) and mod(nf2,block_size_y) should be 0" << endl;
				return 0;
			}
			ier = cnufftspread2d_gpu_odriven(nf1, nf2, fw, M, x, y, c, opts);
			if(ier != 0 ){
				cout<<"error: cnufftspread2d_gpu_odriven"<<endl;
				return 0;
			}
			FLT todriven=timer.elapsedsec();
			printf("[odriven] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
					M,N1,N2,nf1*nf2,todriven,M/todriven);
		}
		break;	
		case 4:
		{
			timer.restart();
			opts.bin_size_x=32;
			opts.bin_size_y=32;
			ier = cnufftspread2d_gpu_hybrid(nf1, nf2, fw, M, x, y, c, opts);
			if(ier != 0 ){
				cout<<"error: cnufftspread2d_gpu_hybrid"<<endl;
				return 0;
			}
			FLT thybrid=timer.elapsedsec();
			printf("[hybrid ] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
					M,N1,N2,nf1*nf2,thybrid,M/thybrid);
		}
		break;
		default:
			cout<<"error: incorrect method, should be 1,2,3 or 4"<<endl;
			return 0;
	}

#ifdef RESULT
	switch(method)
	{
		case 3:
			opts.bin_size_x=4;
			opts.bin_size_y=4;
		case 4:
			opts.bin_size_x=32;
			opts.bin_size_y=32;
	}
	cout<<"[result-input]"<<endl;
	for(int j=0; j<nf2; j++){
		if( j % opts.bin_size_y == 0)
			printf("\n");
		for (int i=0; i<nf1; i++){
			if( i % opts.bin_size_x == 0 && i!=0)
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
