#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>

#include <cufinufft_eitherprec.h>

#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int N1, N2, M1, M2, N;
	if (argc<4) {
		fprintf(stderr,
			"Usage: cufinufft2d1nupts_test method N1 N2\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven,\n"
			"    2: sub-problem, or\n"
			"    3: sub-problem with Paul's idea.\n");
		return 1;
	}
	int method;
	sscanf(argv[1],"%d",&method);

	N1 = 100;
	N2 = 100;
	N = N1*N2;
	M1 = N1*N2;
	M2 = 2*N1*N2;

	FLT tol=1e-5;
	int iflag=1;

	cout<<scientific<<setprecision(3);
	int ier;

	FLT *x1, *y1;
	CPX *c1, *fk1;
	cudaMallocHost(&x1, M1*sizeof(FLT));
	cudaMallocHost(&y1, M1*sizeof(FLT));
	cudaMallocHost(&c1, M1*sizeof(CPX));
	cudaMallocHost(&fk1,N1*N2*sizeof(CPX));

	FLT *d_x1, *d_y1;
	CUCPX *d_c1, *d_fk1;
	checkCudaErrors(cudaMalloc(&d_x1,M1*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y1,M1*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c1,M1*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fk1,N1*N2*sizeof(CUCPX)));

	FLT *x2, *y2;
	CPX *c2, *fk2;
	cudaMallocHost(&x2, M2*sizeof(FLT));
	cudaMallocHost(&y2, M2*sizeof(FLT));
	cudaMallocHost(&c2, M2*sizeof(CPX));
	cudaMallocHost(&fk2,N1*N2*sizeof(CPX));

	FLT *d_x2, *d_y2;
	CUCPX *d_c2, *d_fk2;
	checkCudaErrors(cudaMalloc(&d_x2,M2*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y2,M2*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c2,M2*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fk2,N1*N2*sizeof(CUCPX)));

	// Making data
	for (int i = 0; i < M1; i++) {
		x1[i] = M_PI*randm11();// x in [-pi,pi)
		y1[i] = M_PI*randm11();
		c1[i].real(randm11());
		c1[i].imag(randm11());
	}

	for (int i = 0; i < M2; i++) {
		x2[i] = M_PI*randm11();// x in [-pi,pi)
		y2[i] = M_PI*randm11();
		c2[i].real(randm11());
		c2[i].imag(randm11());
	}

	checkCudaErrors(cudaMemcpy(d_x1,x1,M1*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y1,y1,M1*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c1,c1,M1*sizeof(CUCPX),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_x2,x2,M2*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y2,y2,M2*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c2,c2,M2*sizeof(CUCPX),cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	float milliseconds = 0;
	float totaltime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

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

	// now to our tests...
	CUFINUFFT_PLAN dplan;
	int dim = 2;
	int type = 1;

	// Here we setup our own opts, for gpu_method.
	cufinufft_opts opts;
	ier=CUFINUFFT_DEFAULT_OPTS(type, dim, &opts);
	if(ier!=0){
	  printf("err %d: CUFINUFFT_DEFAULT_OPTS\n", ier);
	  return ier;
	}

	opts.gpu_method=method;

	int nmodes[3];
	int ntransf = 1;
	int maxbatchsize = 1;
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = 1;
	cudaEventRecord(start);
	ier=CUFINUFFT_MAKEPLAN(type, dim, nmodes, iflag, ntransf, tol,
			       maxbatchsize, &dplan, &opts);
	if (ier!=0){
	  printf("err: cufinufft2d_plan\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);


	cudaEventRecord(start);
	ier=CUFINUFFT_SETPTS(M1, d_x1, d_y1, NULL, 0, NULL, NULL, NULL, dplan);
	if (ier!=0){
	  printf("err: cufinufft_setpts (set 1)\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts (set 1):\t %.3g s\n", milliseconds/1000);


	cudaEventRecord(start);
	ier=CUFINUFFT_EXECUTE(d_c1, d_fk1, dplan);
	if (ier!=0){
	  printf("err: cufinufft2d1_exec (set 1)\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	float exec_ms = milliseconds;
	printf("[time  ] cufinufft exec (set 1):\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	ier=CUFINUFFT_SETPTS(M2, d_x2, d_y2, NULL, 0, NULL, NULL, NULL, dplan);
	if (ier!=0){
	  printf("err: cufinufft_setpts (set 2)\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts (set 2):\t %.3g s\n", milliseconds/1000);


	cudaEventRecord(start);
	ier=CUFINUFFT_EXECUTE(d_c2, d_fk2, dplan);
	if (ier!=0){
	  printf("err: cufinufft2d1_exec (set 2)\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	exec_ms += milliseconds;
	printf("[time  ] cufinufft exec (set 2):\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	ier=CUFINUFFT_DESTROY(dplan);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	checkCudaErrors(cudaMemcpy(fk1,d_fk1,N1*N2*sizeof(CUCPX),
		cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fk2,d_fk2,N1*N2*sizeof(CUCPX),
		cudaMemcpyDeviceToHost));

	printf("[Method %d] (%d+%d) NU pts to %d U pts in %.3g s:      %.3g NU pts/s\n",
			opts.gpu_method,M1,M2,N1*N2,totaltime/1000,(M1+M2)/totaltime*1000);
	printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n",(M1+M2)/exec_ms*1000);

	int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2);  // choose some mode index to check
	CPX Ft = CPX(0,0), J = IMA*(FLT)iflag;
	for (BIGINT j=0; j<M1; ++j)
		Ft += c1[j] * exp(J*(nt1*x1[j]+nt2*y1[j]));   // crude direct
	int it = N1/2+nt1 + N1*(N2/2+nt2);   // index in complex F as 1d array
//	printf("[gpu   ] one mode: abs err in F[%ld,%ld] is %.3g\n",(int)nt1,(int)nt2,abs(Ft-fk[it]));
	printf("[gpu   ] one mode: rel err in F[%ld,%ld] is %.3g (set 1)\n",(int)nt1,(int)nt2,abs(Ft-fk1[it])/infnorm(N,fk1));
	Ft = CPX(0,0);
	for (BIGINT j=0; j<M2; ++j)
		Ft += c2[j] * exp(J*(nt1*x2[j]+nt2*y2[j]));   // crude direct
	printf("[gpu   ] one mode: rel err in F[%ld,%ld] is %.3g (set 2)\n",(int)nt1,(int)nt2,abs(Ft-fk2[it])/infnorm(N,fk2));

	cudaFreeHost(x1);
	cudaFreeHost(y1);
	cudaFreeHost(c1);
	cudaFreeHost(fk1);
	cudaFreeHost(x2);
	cudaFreeHost(y2);
	cudaFreeHost(c2);
	cudaFreeHost(fk2);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_c1);
	cudaFree(d_fk1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_c2);
	cudaFree(d_fk2);

	// for cuda-memcheck to work
	cudaDeviceReset();

	return 0;
}
