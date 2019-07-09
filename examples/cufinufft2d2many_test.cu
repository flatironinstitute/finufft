#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>

#include "../src/spreadinterp.h"
#include "../src/cufinufft.h"
#include "../finufft/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	FLT sigma = 2.0;
	int N1, N2, M;
	int ntransf, ntransfcufftplan;
	if (argc<4) {
		fprintf(stderr,"Usage: cufinufft2d2_test [method [N1 N2 [ntransf [ntransfcufftplan [M [tol]]]]\n");
		fprintf(stderr,"Details --\n");
		fprintf(stderr,"method 1: input driven without sorting\n");
		fprintf(stderr,"method 5: subprob\n");
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

	ntransfcufftplan = min(8, ntransf);
	if(argc>5){
		sscanf(argv[5],"%d",&ntransfcufftplan);
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
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, ntransf*M*sizeof(CPX));
	cudaMallocHost(&fk,ntransf*N1*N2*sizeof(CPX));

	// Making data
	for (int i = 0; i < M; i++) {
		x[i] = M_PI*randm11();// x in [-pi,pi)
		y[i] = M_PI*randm11();
	}

	for(int i=0; i<ntransf*N1*N2; i++){
		fk[i].real() = randm11();
		fk[i].imag() = randm11();
	}

	cudaEvent_t start, stop;
	float milliseconds = 0;
	double totaltime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*warm up gpu*/
	cudaEventRecord(start);
	char *a;
	checkCudaErrors(cudaMalloc(&a,1));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tWarm up GPU \t\t %.3g s\n", milliseconds/1000);
#endif

	cufinufft_plan dplan;
	cufinufft_opts opts;
	ier=cufinufft_default_opts(opts,tol,sigma);
	opts.method=method;
	opts.spread_direction=2;

	cudaEventRecord(start);
	ier=cufinufft2d_plan(M, N1, N2, ntransf, ntransfcufftplan, iflag, opts, 
		&dplan);
	if (ier!=0){
		printf("err: cufinufft2d_plan\n");
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	ier=cufinufft2d_setNUpts(x, y, opts, &dplan);
	if (ier!=0){
		printf("err: cufinufft2d_setNUpts\n");
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	ier=cufinufft2d2_exec(c, fk, opts, &dplan);
	if (ier!=0){
		printf("err: cufinufft2d2_exec\n");
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	ier=cufinufft2d_destroy(opts, &dplan);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	// This must be here, since in gpu code, x, y gets modified if pirange=1
#if 0
	CPX* fkstart; 
	CPX* cstart;
	for(int t=0; t<ntransf; t++){
		fkstart = fk + t*N1*N2;
		cstart = c + t*M;
		int jt = M/2;          // check arbitrary choice of one targ pt
		CPX J = IMA*(FLT)iflag;
		CPX ct = CPX(0,0);
		int m=0;
		for (int m2=-(N2/2); m2<=(N2-1)/2; ++m2)  // loop in correct order over F
			for (int m1=-(N1/2); m1<=(N1-1)/2; ++m1)
				ct += fkstart[m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct
		
		printf("[gpu   ] one targ: rel err in c[%ld] is %.3g\n",(int64_t)jt,abs(cstart[jt]-ct)/infnorm(M,c));
	}
#endif
#if 0
	cout<<"[result-input]"<<endl;
	for(int j=0; j<nf2; j++){
		//        if( j % opts.bin_size_y == 0)
		//                printf("\n");
		for (int i=0; i<nf1; i++){
			//                if( i % opts.bin_size_x == 0 && i!=0)
			//                        printf(" |");
			printf(" (%2.3g,%2.3g)",fw[i+j*nf1].real(),fw[i+j*nf1].imag() );
		}
		cout<<endl;
	}
#endif	
	printf("[totaltime] %.3g us, speed %.3g NUpts/s\n", totaltime*1000, M*ntransf/totaltime*1000);
	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fk);
	return 0;
}
