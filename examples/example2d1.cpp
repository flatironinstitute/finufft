#include <iostream>
#include <iomanip>
#include <math.h>
#include <complex>

#include "../src/cufinufft.h"

using namespace std;

int main(int argc, char* argv[])
{
	cout<<scientific<<setprecision(3);

	int ier;
	int N1 = 256;
	int N2 = 256;
	int M = 65536;
	int ntransf = 16;
	int ntransfcufftplan = 8;
	int iflag=1;
	FLT tol=1e-6;

	FLT *x, *y;
	CPX *c, *fk;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, M*ntransf*sizeof(CPX));
	cudaMallocHost(&fk,N1*N2*ntransf*sizeof(CPX));

	for (int i=0; i<M; i++) {
		x[i] = M_PI*randm11();
		y[i] = M_PI*randm11();
	}

	for(int i=0; i<M*ntransf; i++){
		c[i].real() = randm11();
		c[i].imag() = randm11();
	}

	cufinufft_plan dplan;

	int dim = 2;
	int nmodes[3];

	ier=cufinufft_default_opts(type1, dim, dplan.opts);

	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = 1;

	ier=cufinufft_makeplan(type1, dim, nmodes, iflag, ntransf, tol, 
		ntransfcufftplan, &dplan);

	ier=cufinufft_setNUpts(M, x, y, NULL, 0, NULL, NULL, NULL, &dplan);

	ier=cufinufft_exec(c, fk, &dplan);

	ier=cufinufft_destroy(&dplan);


	cout<<endl<<"Accuracy check:"<<endl;
	int N = N1*N2;
	for(int i=0; i<ntransf; i+=5){
		int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2);  // choose some mode index to check
		CPX Ft = CPX(0,0), J = IMA*(FLT)iflag;
		for (BIGINT j=0; j<M; ++j)
			Ft += c[j+i*M] * exp(J*(nt1*x[j]+nt2*y[j]));   // crude direct
		int it = N1/2+nt1 + N1*(N2/2+nt2);   // index in complex F as 1d array
		printf("[gpu %3d] one mode: abs err in F[%ld,%ld] is %.3g\n",i,(int)nt1,
			(int)nt2,abs(Ft-fk[it+i*N]));
		printf("[gpu %3d] one mode: rel err in F[%ld,%ld] is %.3g\n",i,(int)nt1,
			(int)nt2,abs(Ft-fk[it+i*N])/infnorm(N,fk+i*N));
	}

	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fk);
	return 0;
}
