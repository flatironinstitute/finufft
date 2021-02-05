#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "../src/cuspreadinterp.h"
#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1, nf2, nf3;
	FLT sigma = 2.0;
	int N1, N2, N3, M;
	if (argc<6) {
		fprintf(stderr,
			"Usage: spread3d method nupts_distr nf1 nf2 nf3 [maxsubprobsize [M [tol [kerevalmeth [sort]]]]]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven,\n"
			"    2: sub-problem, or\n"
			"    4: block gather (each nf must be multiple of 8).\n"
			"  nupts_distr: The distribution of the points; one of\n"
			"    0: uniform, or\n"
			"    1: concentrated in a small region.\n"
			"  nf1, nf2, nf3: The size of the 3D array.\n"
			"  maxsubprobsize: Maximum size of subproblems (default 65536).\n"
			"  M: The number of non-uniform points (default nf1 * nf2 * nf3 / 8).\n"
			"  tol: NUFFT tolerance (default 1e-6).\n"
			"  kerevalmeth: Kernel evaluation method; one of\n"
			"     0: Exponential of square root (default), or\n"
			"     1: Horner evaluation.\n"
			"  sort: One of\n"
			"     0: do not sort the points, or\n"
			"     1: sort the points (default).\n");
		return 1;
	}
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	int nupts_distribute;
	sscanf(argv[2],"%d",&nupts_distribute);
	sscanf(argv[3],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[4],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!
	sscanf(argv[5],"%lf",&w); nf3 = (int)w;  // so can read 1e6 right!

	int maxsubprobsize=1024;
	if(argc>6){
		sscanf(argv[6],"%d",&maxsubprobsize);
	}
	N1 = (int) nf1/sigma;
	N2 = (int) nf2/sigma;
	N3 = (int) nf3/sigma;
	M = N1*N2*N3;// let density always be 1
	if(argc>7){
		sscanf(argv[7],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
		//if(M == 0) M=N1*N2;
	}

	FLT tol=1e-6;
	if(argc>8){
		sscanf(argv[8],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}

	int kerevalmeth=0;
	if(argc>9){
		sscanf(argv[9],"%d",&kerevalmeth);
	}

	int sort=1;
	if(argc>10){
		sscanf(argv[10],"%d",&sort);
	}

	int ier;
	FLT *x, *y, *z;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&z, M*sizeof(FLT));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fw,nf1*nf2*nf3*sizeof(CPX));

	FLT *d_x, *d_y, *d_z;
	CUCPX *d_c, *d_fw;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_z,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fw,nf1*nf2*nf3*sizeof(CUCPX)));

	int dim=3;
	CUFINUFFT_PLAN dplan = new CUFINUFFT_PLAN_S;
	// Zero out your struct, (sets all pointers to NULL, crucial)
        memset(dplan, 0, sizeof(*dplan));
	ier = CUFINUFFT_DEFAULT_OPTS(1, dim, &(dplan->opts));

	dplan->opts.gpu_method          =method;
	dplan->opts.gpu_maxsubprobsize  =maxsubprobsize;
	dplan->opts.gpu_kerevalmeth     =kerevalmeth;
	dplan->opts.gpu_sort            =sort;
	dplan->opts.gpu_spreadinterponly=1;
	ier = setup_spreader_for_nufft(dplan->spopts, tol, dplan->opts);

	//binsize, obinsize need to be set here, since SETUP_BINSIZE() is not 
	//called in spread, interp only wrappers.
	if(dplan->opts.gpu_method == 4)
	{
		dplan->opts.gpu_binsizex=4;
		dplan->opts.gpu_binsizey=4;
		dplan->opts.gpu_binsizez=4;
		dplan->opts.gpu_obinsizex=8;
		dplan->opts.gpu_obinsizey=8;
		dplan->opts.gpu_obinsizez=8;
		dplan->opts.gpu_maxsubprobsize=maxsubprobsize;
	}
	if(dplan->opts.gpu_method == 2)
	{
		dplan->opts.gpu_binsizex=16;
		dplan->opts.gpu_binsizey=16;
		dplan->opts.gpu_binsizez=2;
		dplan->opts.gpu_maxsubprobsize=maxsubprobsize;
	}
	if(dplan->opts.gpu_method == 1)
	{
		dplan->opts.gpu_binsizex=16;
		dplan->opts.gpu_binsizey=16;
		dplan->opts.gpu_binsizez=2;
	}

	cout<<scientific<<setprecision(3);

	switch(nupts_distribute){
		// Making data
		case 0: //uniform
			{
				for (int i = 0; i < M; i++) {
					x[i] = M_PI*randm11();
					y[i] = M_PI*randm11();
					z[i] = M_PI*randm11();
					c[i].real(randm11());
					c[i].imag(randm11());
				}
			}
			break;
		case 1: // concentrate on a small region
			{
				for (int i = 0; i < M; i++) {
					x[i] = M_PI*rand01()/nf1*16;
					y[i] = M_PI*rand01()/nf2*16;
					z[i] = M_PI*rand01()/nf3*16;
					c[i].real(randm11());
					c[i].imag(randm11());
				}
			}
			break;
		default:
			cerr << "not valid nupts distr" << endl;
			return 1;
	}

	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_z,z,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c,c,M*sizeof(CUCPX),cudaMemcpyHostToDevice));

	CNTime timer;
	/*warm up gpu*/
	char *a;
	timer.restart();
	checkCudaErrors(cudaMalloc(&a,1));
	// cout<<"[time  ]"<< " (warm up) First cudamalloc call " << timer.elapsedsec()
	//	<<" s"<<endl<<endl;



	timer.restart();
	ier = CUFINUFFT_SPREAD3D(nf1, nf2, nf3, d_fw, M, d_x, d_y, d_z, d_c, dplan);
	if(ier != 0 ){
		cout<<"error: cnufftspread3d"<<endl;
		return 0;
	}
	FLT t=timer.elapsedsec();
	printf("[Method %d] %ld NU pts to #%d U pts in %.3g s (%.3g NU pts/s)\n",
			dplan->opts.gpu_method,M,nf1*nf2*nf3,t,M/t);
#ifdef RESULT
	cout<<"[result-input]"<<endl;
	for(int k=0; k<nf3; k++){
		for(int j=0; j<nf2; j++){
			for (int i=0; i<nf1; i++){
				if( i % dplan.opts.gpu_binsizex == 0 && i!=0)
					printf(" |");
				printf(" (%2.3g,%2.3g)",fw[i+j*nf1+k*nf2*nf1].real(),
					fw[i+j*nf1+k*nf2*nf1].imag() );
			}
			cout<<endl;
		}
		cout<<"----------------------------------------------------------------"<<endl;
	}
#endif

	cudaDeviceReset();
	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(z);
	cudaFreeHost(c);
	cudaFreeHost(fw);
	return 0;
}
