#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "../src/spread.h"
#include "../src/finufft/utils.h"
#include "../src/finufft/cnufftspread.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1, nf2;
	FLT sigma = 2.0;
	int N1, N2, M;
	if (argc<4) {
		fprintf(stderr,"Usage: accuracy [nupts_distribute [N1 N2 [M [tol]]]]\n");
		return 1;
	}  
	int nupts_distribute;
	sscanf(argv[1],"%d",&nupts_distribute);

	double w;
	sscanf(argv[2],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[3],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!

	N1 = (int) nf1/sigma;
	N2 = (int) nf2/sigma;
	M = N1*N2;// let density always be 1
	if(argc>4){
		sscanf(argv[4],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	FLT tol=1e-6;
	if(argc>5){
		sscanf(argv[5],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}


	int ns=std::ceil(-log10(tol/10.0));
	spread_opts opts;
	opts.nspread=ns;
	opts.upsampfac=2.0;
	opts.ES_beta= 2.30 * (FLT)ns;
	opts.ES_c=4.0/(ns*ns);
	opts.ES_halfwidth=(FLT)ns/2;
	opts.Horner=0;
	opts.maxsubprobsize=1000;
	opts.pirange=0;
	opts.bin_sort=1;

	cout<<scientific<<setprecision(3);
	int ier;


	FLT *x, *y;
	CPX *c, *fwic, *fwi, *fwo, *fwh, *fws, *fwfinufft;
	cudaMallocHost(&x, M*sizeof(CPX));
	cudaMallocHost(&y, M*sizeof(CPX));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fwi,       nf1*nf2*sizeof(CPX));
	cudaMallocHost(&fwic,      nf1*nf2*sizeof(CPX));
	cudaMallocHost(&fwo,       nf1*nf2*sizeof(CPX));
	cudaMallocHost(&fwh,       nf1*nf2*sizeof(CPX));
	cudaMallocHost(&fws,       nf1*nf2*sizeof(CPX));
	cudaMallocHost(&fwfinufft, nf1*nf2*sizeof(CPX));

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


	/* -------------------------------------- */
	// Method 1: Input driven without sorting //
	/* -------------------------------------- */
	timer.restart();
	opts.method=1;
	ier = cnufftspread2d_gpu(nf1, nf2, fwi, M, x, y, c, opts);
	if(ier != 0 ){
		cout<<"error: cnufftspread2d_gpu_idriven"<<endl;
		return 0;
	}
	FLT tidriven=timer.elapsedsec();
	printf("[idriven] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
			M,N1,N2,nf1*nf2,tidriven,M/tidriven);

	/* -------------------------------------- */
	// Method 2: Input driven with sorting    //
	/* -------------------------------------- */
	timer.restart();
	opts.method=2;
	opts.bin_size_x=16;
	opts.bin_size_y=16;
	ier = cnufftspread2d_gpu(nf1, nf2, fwic, M, x, y, c, opts);
	FLT ticdriven=timer.elapsedsec();
	printf("[isorted] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
			M,N1,N2,nf1*nf2,ticdriven,M/ticdriven);
	
	/* -------------------------------------- */
	// Method 3: Output driven                //
	/* -------------------------------------- */
	if(nupts_distribute == 1){
		timer.restart();
		opts.method=3;
		opts.bin_size_x=4;
		opts.bin_size_y=4;
		ier = cnufftspread2d_gpu(nf1, nf2, fwo, M, x, y, c, opts);
		if(ier != 0 ){
			cout<<"error: cnufftspread2d_gpu_odriven"<<endl;
			return 0;
		}
		FLT todriven=timer.elapsedsec();
		printf("[odriven] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
			M,N1,N2,nf1*nf2,todriven,M/todriven);
	}

	/* -------------------------------------- */
	// Method 4: Hybrid                       //
	/* -------------------------------------- */
	timer.restart();
	opts.method=4;
	opts.bin_size_x=32;
	opts.bin_size_y=32;
	ier = cnufftspread2d_gpu(nf1, nf2, fwh, M, x, y, c, opts);
	FLT thybrid=timer.elapsedsec();
	if(ier != 0 ){
		cout<<"error: cnufftspread2d_gpu_hybrid"<<endl;
		return 0;
	}
	printf("[hybrid ] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
			M,N1,N2,nf1*nf2,thybrid,M/thybrid);

	/* -------------------------------------- */
	// Method 5: Subprob                     //
	/* -------------------------------------- */
	timer.restart();
	opts.method=5;
	opts.bin_size_x=32;
	opts.bin_size_y=32;
	ier = cnufftspread2d_gpu(nf1, nf2, fws, M, x, y, c, opts);
	FLT tsubprob=timer.elapsedsec();
	if(ier != 0 ){
		cout<<"error: cnufftspread2d_gpu_subprob"<<endl;
		return 0;
	}
	printf("[subprob ] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
			 M,N1,N2,nf1*nf2,tsubprob,M/thybrid);
	/* -------------------------------------- */
	// FINUTFFT cpu spreader                  //
	/* -------------------------------------- */
	timer.start();
	setup_spreader(opts,(FLT)tol,opts.upsampfac,opts.kerevalmeth);
	opts.pirange=0;
	opts.chkbnds=1;
	opts.spread_direction=1;
	opts.flags=0;//ker always return 1
	opts.kerevalmeth=1;
	opts.kerpad=1;
	opts.sort_threads=0;
	opts.sort=2;
	opts.debug=0;

	ier = cnufftspread(nf1,nf2,1,(FLT*) fwfinufft,M,x,y,NULL,(FLT*) c,opts);
	FLT t=timer.elapsedsec();
	if (ier!=0) {
		printf("error (ier=%d)!\n",ier);
		return ier;
	}
	printf("[finufft] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
		M,N1,N2,nf1*nf2,t,M/t);
		//printf("    %.3g NU pts in %.3g s \t%.3g pts/s \t%.3g spread pts/s\n",(double)M,t,M/t,pow(opts.nspread,2)*M/t);
	/* ------------------------------------------------------------------------------------------------------*/
	
	cout<<endl;
	FLT err=relerrtwonorm(nf1*nf2,fwi,fwfinufft);
	printf("|| fwi  - fwfinufft ||_2 / || fwi  ||_2 =  %.6g\n", err);
	err=relerrtwonorm(nf1*nf2,fwic,fwfinufft);
	printf("|| fwic - fwfinufft ||_2 / || fwic ||_2 =  %.6g\n", err);
	if(nupts_distribute == 1){
		err=relerrtwonorm(nf1*nf2,fwo,fwfinufft);
		printf("|| fwo  - fwfinufft ||_2 / || fwo  ||_2 =  %.6g\n", err);
	}
	err=relerrtwonorm(nf1*nf2,fwh,fwfinufft);
	printf("|| fwh  - fwfinufft ||_2 / || fwh  ||_2 =  %.6g\n", err);
	err=relerrtwonorm(nf1*nf2,fws,fwfinufft);
	printf("|| fws  - fwfinufft ||_2 / || fwh  ||_2 =  %.6g\n", err);

#ifdef RESULT
	cout<<"[resultdiff]"<<endl;
	FLT fwi_infnorm=infnorm(nf1*nf2, fwi);
	int nn=0;
	for(int j=0; j<nf2; j++){
		for (int i=0; i<nf1; i++){
			if( norm(fwi[i+j*nf1]-fwh[i+j*nf1])/fwi_infnorm > 1e-5 & nn<10){
				cout<<norm(fwi[i+j*nf1]-fwh[i+j*nf1])/fwi_infnorm<<" ";
				cout<<"(i,j)=("<<i<<","<<j<<"), "<<fwi[i+j*nf1] <<","<<fwh[i+j*nf1]<<endl;
				nn++;
			}
		}
	}
	cout<<endl;
#endif
#ifdef RESULT
	cout<<"[result-hybrid]"<<endl;
	for(int j=0; j<nf2; j++){
		if( j % opts.bin_size_y == 0)
			printf("\n");
		for (int i=0; i<nf1; i++){
			if( i % opts.bin_size_x == 0 && i!=0)
				printf(" |");
			printf(" (%2.3g,%2.3g)",fwh[i+j*nf1].real(),fwh[i+j*nf1].imag() );
			//cout<<" "<<setw(8)<<fwo[i+j*nf1];
		}
		cout<<endl;
	}
	cout<<endl;
#endif

	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fwi);
	cudaFreeHost(fwic);
	//cudaFreeHost(fwo);
	cudaFreeHost(fwh);
	cudaFreeHost(fwfinufft);
	return 0;
}
