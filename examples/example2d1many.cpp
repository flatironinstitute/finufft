#include <iostream>
#include <iomanip>
#include <math.h>
#include <complex>

#include <cufinufft.h>

using namespace std;

int main(int argc, char* argv[])
/*
 * example code for 2D Type 1 transformation.
 *
 * To compile the code:
 * 	nvcc -DSINGLE example2d1many.cpp -o example2d1 /loc/to/cufinufft/lib-static/libcufinufftf.a -lcudart -lcufft -lnvToolsExt
 * 
 * or
 * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/loc/to/cufinufft/lib
 * nvcc -DSINGLE example2d1many.cpp -L/loc/to/cufinufft/lib/ -o example2d1 -lcufinufftf
 *
 *
 */
{
	cout<<scientific<<setprecision(3);

	int ier;
	int N1 = 256;
	int N2 = 256;
	int M = 65536;
	int ntransf = 1;
	int maxbatchsize = 1;
	int iflag=1;
	FLT tol=1e-6;

	FLT *x, *y;
	CPX *c, *fk;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, M*ntransf*sizeof(CPX));
	cudaMallocHost(&fk,N1*N2*ntransf*sizeof(CPX));

	FLT *d_x, *d_y;
	CUCPX *d_c, *d_fk;
	cudaMalloc(&d_x,M*sizeof(FLT));
	cudaMalloc(&d_y,M*sizeof(FLT));
	cudaMalloc(&d_c,M*ntransf*sizeof(CUCPX));
	cudaMalloc(&d_fk,N1*N2*ntransf*sizeof(CUCPX));

	for (int i=0; i<M; i++) {
		x[i] = M_PI*randm11();
		y[i] = M_PI*randm11();
	}

	for(int i=0; i<M*ntransf; i++){
		c[i].real(randm11());
		c[i].imag(randm11());
	}
	cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice);
	cudaMemcpy(d_c,c,M*ntransf*sizeof(CUCPX),cudaMemcpyHostToDevice);

	cufinufft_plan dplan;

	int dim = 2;
	int nmodes[3];

	ier=cufinufft_default_opts(type1, dim, &dplan.opts);

	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = 1;

	ier=cufinufft_makeplan(type1, dim, nmodes, iflag, ntransf, tol, 
		maxbatchsize, &dplan);

	ier=cufinufft_setNUpts(M, d_x, d_y, NULL, 0, NULL, NULL, NULL, &dplan);

	ier=cufinufft_exec(d_c, d_fk, &dplan);

	ier=cufinufft_destroy(&dplan);

	cudaMemcpy(fk,d_fk,N1*N2*ntransf*sizeof(CUCPX),cudaMemcpyDeviceToHost);

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

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_c);
	cudaFree(d_fk);
	return 0;
}
