#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>

#include <cufinufft.h>
#include <profile.h>
#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int N1, N2, M;
	if (argc<4) {
		fprintf(stderr,
			"Usage: cufinufft2d2_test method N1 N2 [M [tol]]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven, or\n"
			"    2: sub-problem.\n"
			"  N1, N2: The size of the 2D array.\n"
			"  M: The number of non-uniform points (default N1 * N2).\n"
			"  tol: NUFFT tolerance (default 1e-6).\n");
		return 1;
	}  
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	sscanf(argv[2],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[3],"%lf",&w); N2 = (int)w;  // so can read 1e6 right!
	M = N1*N2;// let density always be 1
	if(argc>4){
		sscanf(argv[4],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	FLT tol=1e-6;
	if(argc>5){
		sscanf(argv[5],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}
	int iflag=1;


	cout<<scientific<<setprecision(3);
	int ier;


	FLT *x, *y;
	CPX *c, *fk;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fk,N1*N2*sizeof(CPX));

	FLT *d_x, *d_y;
	CUCPX *d_c, *d_fk;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fk,N1*N2*sizeof(CUCPX)));
	// Making data
	for (int i = 0; i < M; i++) {
		x[i] = M_PI*randm11();// x in [-pi,pi)
		y[i] = M_PI*randm11();
	}
	for(int i=0; i<N1*N2; i++){
		fk[i].real(1.0);
		fk[i].imag(1.0);
	}
	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fk, fk, N1*N2*sizeof(CPX), 
		cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*warm up gpu*/
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("Warm Up",1);
		char *a;
		checkCudaErrors(cudaMalloc(&a,1));
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tWarm up GPU \t\t %.3g s\n", milliseconds/1000);
#endif

	cufinufft_plan dplan;
	int dim = 2;
	int type = 2;
	ier=cufinufft_default_opts(type, dim, dplan.opts);
	dplan.opts.gpu_method=method;

	int nmodes[3];
	int ntransf = 1;
	int ntransfcufftplan = 1;
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = 1;
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_plan",2);
		ier=cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, 
			ntransfcufftplan, &dplan);
		if (ier!=0){
			printf("err: cufinufft2d_plan\n");
		}
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);
#endif
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_setNUpts",3);
		ier=cufinufft_setNUpts(M, d_x, d_y, NULL, 0, NULL, NULL, NULL, &dplan);
		if (ier!=0){
			printf("err: cufinufft_setNUpts\n");
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d2_exec",4);
		ier=cufinufft_exec(d_c, d_fk, &dplan);
		if (ier!=0){
			printf("err: cufinufft2d2_exec\n");
		}
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);
#endif
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_destroy",5);
		ier=cufinufft_destroy(&dplan);
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);
#endif
	checkCudaErrors(cudaMemcpy(c,d_c,M*sizeof(CUCPX),cudaMemcpyDeviceToHost));
	int jt = M/2;          // check arbitrary choice of one targ pt
	CPX J = IMA*(FLT)iflag;
	CPX ct = CPX(0,0);
	int m=0;
	for (int m2=-(N2/2); m2<=(N2-1)/2; ++m2)  // loop in correct order over F
		for (int m1=-(N1/2); m1<=(N1-1)/2; ++m1)
			ct += fk[m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct
	printf("[gpu   ] one targ: rel err in c[%ld] is %.3g\n",(int64_t)jt,abs(c[jt]-ct)/infnorm(M,c));
#if 0
	cout<<"[result-input]"<<endl;
	for(int j=0; j<nf2; j++){
		//        if( j % opts.gpu_binsizey == 0)
		//                printf("\n");
		for (int i=0; i<nf1; i++){
			//                if( i % opts.gpu_binsizex == 0 && i!=0)
			//                        printf(" |");
			printf(" (%2.3g,%2.3g)",fw[i+j*nf1].real(),fw[i+j*nf1].imag() );
		}
		cout<<endl;
	}
#endif	
	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fk);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_c);
	cudaFree(d_fk);
	return 0;
}
