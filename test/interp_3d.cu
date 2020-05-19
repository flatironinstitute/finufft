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
	if (argc<5) {
		fprintf(stderr,"Usage: interp3d [method [nupts_distr [nf1 nf2 nf3 [M [tol [sort]]]]]]\n");
		fprintf(stderr,"Details --\n");
		fprintf(stderr,"method 1: nupts driven\n");
		fprintf(stderr,"method 2: sub-problem\n");
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

	N1 = (int) nf1/sigma;
	N2 = (int) nf2/sigma;
	N3 = (int) nf3/sigma;
	M = N1*N2*N3;// let density always be 1
	if(argc>6){
		sscanf(argv[6],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
		if(M == 0) M=N1*N2*N3;
	}

	FLT tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}

	int sort=1;
	if(argc>8){
		sscanf(argv[8],"%d",&sort);
	}
	int ier;

	int ns=std::ceil(-log10(tol/10.0));
	int maxsubprobsize;

	cout<<scientific<<setprecision(3);


	FLT *x, *y, *z;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&z, M*sizeof(FLT));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fw,nf1*nf2*nf3*sizeof(CPX));

	switch(nupts_distribute){
		// Making data
		case 1: //uniform
			{
				for (int i = 0; i < M; i++) {
					x[i] = RESCALE(M_PI*randm11(), nf1, 1);// x in [-pi,pi)
					y[i] = RESCALE(M_PI*randm11(), nf2, 1);
					z[i] = RESCALE(M_PI*randm11(), nf3, 1);
					//cout << x[i] << "," << y[i] << "," << z[i] << endl;
				}
				maxsubprobsize = 65536;
			}
			break;
		case 2: // concentrate on a small region
			{
				for (int i = 0; i < M; i++) {
					x[i] = RESCALE(M_PI*rand01()/(nf1*2/32), nf1, 1);// x in [-pi,pi)
					y[i] = RESCALE(M_PI*rand01()/(nf1*2/32), nf2, 1);
					z[i] = RESCALE(M_PI*rand01()/(nf3*2/32), nf3, 1);
				}
				maxsubprobsize = 1024;
			}
			break;
		default:
			cerr<<"error: nupts distr should be 1,2" << endl;
			return 1;
	}
	for(int i=0; i<nf1*nf2*nf3; i++){
		fw[i].real(1.0);
		fw[i].imag(0.0);
	}

	int dim=3;
	cufinufft_plan dplan;
	ier = cufinufft_default_opts(type2, dim, dplan.opts);
	if(ier != 0 ){
		cout<<"error: cufinufft_default_opts"<<endl;
		return 0;
	}
	ier = setup_spreader_for_nufft(dplan.spopts, tol, dplan.opts);

	dplan.opts.upsampfac=sigma;
	dplan.opts.gpu_method=method;
	dplan.opts.gpu_kerevalmeth=1;
	dplan.opts.gpu_sort=sort;
	dplan.spopts.pirange=0;
	if(dplan.opts.gpu_method == 2)
	{
		dplan.opts.gpu_binsizex=16;
		dplan.opts.gpu_binsizey=16;
		dplan.opts.gpu_binsizez=2;
		dplan.opts.gpu_maxsubprobsize=maxsubprobsize;
	}
	if(dplan.opts.gpu_method == 1)
	{
		dplan.opts.gpu_binsizex=16;
		dplan.opts.gpu_binsizey=8;
		dplan.opts.gpu_binsizez=4;
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
	cout<<"[info  ] Interpolating  ["<<nf1<<"x"<<nf2<<"x"<<nf3<<
		"] uniform points to "<<M<<"nupts"<<endl;
#endif
	timer.restart();
	ier = cufinufft_interp3d(N1, N2, N3, nf1, nf2, nf3, fw, M, x, y, z, c, tol,
		&dplan);
	if(ier != 0 ){
		cout<<"error: cnufftinterp3d"<<endl;
		return 0;
	}
	FLT t=timer.elapsedsec();
	printf("[Method %d] %ld U pts to #%d NU pts in %.3g s (\t%.3g U pts/s)\n",
			dplan.opts.gpu_method,nf1*nf2*nf3,M,t,M/t);
#ifdef RESULT
	cout<<"[result-input]"<<endl;
	for(int j=0; j<10; j++){
		printf(" (%2.3g,%2.3g)",c[j].real(),c[j].imag() );
		cout<<endl;
	}
	cout<<endl;
#endif

	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(z);
	cudaFreeHost(c);
	cudaFreeHost(fw);
	return 0;
}
