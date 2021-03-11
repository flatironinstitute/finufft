#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <algorithm>
#include "../src/cuspreadinterp.h"
#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1, nf2;
	FLT upsampfac=2.0;
	int N1, N2, M;
	if (argc<5) {
		fprintf(stderr,
			"Usage: spread2d method nupts_distr nf1 nf2 [maxsubprobsize [M [tol [kerevalmeth]]]]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven,\n"
			"    2: sub-problem, or\n"
			"    3: sub-problem with Paul's idea.\n"
			"  nupts_distr: The distribution of the points; one of\n"
			"    0: uniform, or\n"
			"    1: concentrated in a small region.\n"
			"  nf1, nf2: The size of the 2D array.\n"
			"  maxsubprobsize: Maximum size of subproblems (default 65536).\n"
			"  M: The number of non-uniform points (default nf1 * nf2 / 4).\n"
			"  tol: NUFFT tolerance (default 1e-6).\n"
			"  kerevalmeth: Kernel evaluation method; one of\n"
			"     0: Exponential of square root (default), or\n"
			"     1: Horner evaluation.\n");
		return 1;
	}
	double w;
	int method;
	sscanf(argv[1],"%d",&method);

	int nupts_distribute;
	sscanf(argv[2],"%d",&nupts_distribute);
	sscanf(argv[3],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[4],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!

	int maxsubprobsize=65536;
	if(argc>5){
		sscanf(argv[5],"%d",&maxsubprobsize);
	}

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

	CUFINUFFT_PLAN dplan = new CUFINUFFT_PLAN_S;
        // Zero out your struct, (sets all pointers to NULL, crucial)
        memset(dplan, 0, sizeof(*dplan));
	ier = CUFINUFFT_DEFAULT_OPTS(2, dim, &(dplan->opts));

	dplan->opts.gpu_method           = method;
	dplan->opts.gpu_maxsubprobsize   = maxsubprobsize;
	dplan->opts.gpu_kerevalmeth      = kerevalmeth;
	dplan->opts.gpu_sort             = 1;   // ahb changed from 0
	dplan->opts.gpu_spreadinterponly = 1;
	dplan->opts.gpu_binsizex         = 32; //binsize needs to be set here, since
                                           //SETUP_BINSIZE() is not called in 
                                           //spread, interp only wrappers.
	dplan->opts.gpu_binsizey         = 32;
	ier = setup_spreader_for_nufft(dplan->spopts, tol, dplan->opts);

	cout<<scientific<<setprecision(3);

	FLT *x, *y;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fw,nf1*nf2*sizeof(CPX));

	FLT *d_x, *d_y;
	CUCPX *d_c, *d_fw;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fw,nf1*nf2*sizeof(CUCPX)));

	switch(nupts_distribute){
		// Making data
		case 0: //uniform
			{
				for (int i = 0; i < M; i++) {
					x[i] = M_PI*randm11();// x in [-pi,pi)
					y[i] = M_PI*randm11();
					c[i].real(randm11());
					c[i].imag(randm11());
				}
			}
			break;
		case 1: // concentrate on a small region
			{
				for (int i = 0; i < M; i++) {
					x[i] = M_PI*rand01()/(nf1*2/32);
					y[i] = M_PI*rand01()/(nf2*2/32);
					c[i].real(randm11());
					c[i].imag(randm11());
				}
			}
			break;
		default:
			cerr << "not valid nupts distr" << endl;
	}

	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c,c,M*sizeof(CUCPX),cudaMemcpyHostToDevice));

	CNTime timer;
	/*warm up gpu*/
	timer.restart();
	ier = CUFINUFFT_SPREAD2D(nf1, nf2, d_fw, M, d_x, d_y, d_c, dplan);
	if(ier != 0 ){
		cout<<"error: cnufftspread2d"<<endl;
		return 0;
	}
	FLT t=timer.elapsedsec();
	printf("[Method %d] %ld NU pts to #%d U pts in %.3g s (%.3g NU pts/s)\n",
			dplan->opts.gpu_method,M,nf1*nf2,t,M/t);

	checkCudaErrors(cudaMemcpy(fw,d_fw,nf1*nf2*sizeof(CUCPX),
		cudaMemcpyDeviceToHost));
#ifdef RESULT
	cout<<"[result-input]"<<endl;
	for(int j=0; j<nf2; j++){
		if( j % dplan->opts.gpu_binsizey == 0)
			printf("\n");
		for (int i=0; i<nf1; i++){
			if( i % dplan->opts.gpu_binsizex == 0 && i!=0)
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
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_c);
	cudaFree(d_fw);
	return 0;
}
