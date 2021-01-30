#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>

#include <cufinufft_eitherprec.h>

#include <profile.h>
#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int N1, N2, M;
	int ntransf, maxbatchsize;
	if (argc<4) {
		fprintf(stderr,
			"Usage: cufinufft2d2many_test method N1 N2 [ntransf [maxbatchsize [M [tol]]]]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven, or\n"
			"    2: sub-problem.\n"
			"  N1, N2: The size of the 2D array.\n"
			"  ntransf: Number of inputs (default 2 ^ 27 / (N1 * N2)).\n"
			"  maxbatchsize: Number of simultaneous transforms (or 0 for default).\n"
			"  M: The number of non-uniform points (default N1 * N2).\n"
			"  tol: NUFFT tolerance (default 1e-6).\n");
		return 1;
	}
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	sscanf(argv[2],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[3],"%lf",&w); N2 = (int)w;  // so can read 1e6 right!
	M = 2*N1*N2;// let density always be 2
	ntransf = pow(2,28)/M;
	if(argc>4){
		sscanf(argv[4],"%d",&ntransf);
	}

	maxbatchsize = 0;           // have cufinufft choose the default
	if(argc>5){
		sscanf(argv[5],"%d",&maxbatchsize);
	}

	if(argc>6){
		sscanf(argv[6],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	FLT tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}
	int iflag=1;



	cout<<scientific<<setprecision(3);
	int ier;

	printf("#modes = %d, #inputs = %d, #NUpts = %d\n", N1*N2, ntransf, M);

	FLT *x, *y;
	CPX *c, *fk;
#if 1
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, ntransf*M*sizeof(CPX));
	cudaMallocHost(&fk,ntransf*N1*N2*sizeof(CPX));
#else
	x = (FLT*) malloc(M*sizeof(FLT));
	y = (FLT*) malloc(M*sizeof(FLT));
	c = (CPX*) malloc(ntransf*M*sizeof(CPX));
	fk = (CPX*) malloc(ntransf*N1*N2*sizeof(CPX));
#endif
	FLT *d_x, *d_y;
	CUCPX *d_c, *d_fk;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,ntransf*M*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fk,ntransf*N1*N2*sizeof(CUCPX)));

	// Making data
	for (int i = 0; i < M; i++) {
		x[i] = M_PI*randm11();// x in [-pi,pi)
		y[i] = M_PI*randm11();
	}

	for(int i=0; i<ntransf*N1*N2; i++){
		fk[i].real(randm11());
		fk[i].imag(randm11());
	}

	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fk,fk,N1*N2*ntransf*sizeof(CUCPX),cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float milliseconds = 0;
	double totaltime = 0;

    // warm up CUFFT (is slow, takes around 0.2 sec... )
	cudaEventRecord(start);
	{
		int nf1=1;
		cufftHandle fftplan;
		cufftPlan1d(&fftplan,nf1,CUFFT_TYPE,1);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] dummy warmup call to CUFFT\t %.3g s\n", milliseconds/1000);

        // now to the test...
	CUFINUFFT_PLAN dplan;
	int dim = 2;
	int type = 2;

	// Here we setup our own opts, for gpu_method and gpu_kerevalmeth
	cufinufft_opts opts;
	ier=CUFINUFFT_DEFAULT_OPTS(type, dim, &opts);
	if(ier!=0){
	  printf("err %d: CUFINUFFT_DEFAULT_OPTS\n", ier);
	  return ier;
	}
	opts.gpu_method=method;
	opts.gpu_kerevalmeth=1;

	int nmodes[3];
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = 1;
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_plan",2);
		ier=CUFINUFFT_MAKEPLAN(type, dim, nmodes, iflag, ntransf, tol,
				       maxbatchsize, &dplan, &opts);
		if (ier!=0){
		  printf("err: cufinufft2d_plan\n");
		  return ier;
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_setNUpts",3);
		ier=CUFINUFFT_SETPTS(M, d_x, d_y, NULL, 0, NULL, NULL, NULL, dplan);
		if (ier!=0){
		  printf("err: cufinufft2d_setNUpts\n");
		  return ier;
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_exec",4);
		ier=CUFINUFFT_EXECUTE(d_c, d_fk, dplan);
		if (ier!=0){
		  printf("err: cufinufft2d2_exec\n");
		  return ier;
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	float exec_ms =	milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_destroy",5);
		ier=CUFINUFFT_DESTROY(dplan);
		if(ier!=0){
		  printf("err %d: cufinufft2d2_destroy\n", ier);
		  return ier;
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	checkCudaErrors(cudaMemcpy(c,d_c,M*ntransf*sizeof(CUCPX),cudaMemcpyDeviceToHost));

	CPX* fkstart;
	CPX* cstart;
	int t = ntransf-1;
	fkstart = fk + t*N1*N2;
	cstart = c + t*M;
	int jt = M/2;          // check arbitrary choice of one targ pt
	CPX J = IMA*(FLT)iflag;
	CPX ct = CPX(0,0);
	int m=0;
	for (int m2=-(N2/2); m2<=(N2-1)/2; ++m2)  // loop in correct order over F
		for (int m1=-(N1/2); m1<=(N1-1)/2; ++m1)
			ct += fkstart[m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct

	printf("[gpu   ] %dth data one targ: rel err in c[%ld] is %.3g\n",(int)t, (int64_t)jt,abs(cstart[jt]-ct)/infnorm(M,c));

	printf("[totaltime] %.3g us, speed %.3g NUpts/s\n", totaltime*1000, M*ntransf/totaltime*1000);
        printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n",M*ntransf/exec_ms*1000);


	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fk);
	return 0;
}
