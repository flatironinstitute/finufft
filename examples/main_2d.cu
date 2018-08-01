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
		fprintf(stderr,"Usage: spread2d [method [nupts_distr [N1 N2 [M [tol [use_thrust]]]]]]\n");
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
	int nupts_distribute;
	sscanf(argv[2],"%d",&nupts_distribute);
	sscanf(argv[3],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[4],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!

	N1 = (int) nf1/sigma;
	N2 = (int) nf2/sigma;
	M = N1*N2;// let density always be 1
	if(argc>5){
		sscanf(argv[5],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	FLT tol=1e-6;
	if(argc>6){
		sscanf(argv[6],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}

	int use_thrust=0;
	if(argc>7){
		sscanf(argv[7],"%d",&use_thrust);
	}

	int ns=std::ceil(-log10(tol/10.0));
	spread_opts opts;
	opts.nspread=ns;
	opts.upsampfac=2.0;
	opts.ES_beta=2.30*(FLT)ns;
	opts.ES_c=4.0/(ns*ns);
	opts.ES_halfwidth=(FLT)ns/2;
	opts.use_thrust=use_thrust;
	opts.method=method;
	opts.Horner=0;
	opts.pirange=0;

	cout<<scientific<<setprecision(3);
	int ier;


	FLT *x, *y;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(CPX));
	cudaMallocHost(&y, M*sizeof(CPX));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fw,nf1*nf2*sizeof(CPX));

        switch(nupts_distribute){
                // Making data
                case 1: //uniform
                {
                        for (int i = 0; i < M; i++) {
                                x[i] = RESCALE(M_PI*randm11(), nf1, 1);// x in [-pi,pi)
                                y[i] = RESCALE(M_PI*randm11(), nf2, 1);
                                c[i].real() = randm11();
                                c[i].imag() = randm11();
                        }
                }
                break;
                case 2: // concentrate on a small region
                {
                        for (int i = 0; i < M; i++) {
                                x[i] = RESCALE(M_PI*rand01()/(nf1*2/32), nf1, 1);// x in [-pi,pi)
                                y[i] = RESCALE(M_PI*rand01()/(nf1*2/32), nf2, 1);
                                c[i].real() = randm11();
                                c[i].imag() = randm11();
                        }
                        ofstream xyfile ("nupts_location_nonuniform.txt");

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
	if(opts.method == 3)
	{
		opts.bin_size_x=4;
		opts.bin_size_y=4;
	}

	if(opts.method == 4)
	{
		opts.bin_size_x=32;
		opts.bin_size_y=32;
	}

	timer.restart();
	ier = cnufftspread2d_gpu(nf1, nf2, fw, M, x, y, c, opts);
	if(ier != 0 ){
		cout<<"error: cnufftspread2d"<<endl;
		return 0;
	}
	FLT t=timer.elapsedsec();
	printf("[Method %d] %ld NU pts to #%d U pts in %.3g s (\t%.3g NU pts/s)\n",
		opts.method,M,nf1*nf2,t,M/t);
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
