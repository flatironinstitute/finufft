#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include "../src/spreadinterp.h"
#include "../finufft/utils.h"
#include "../finufft/spreadinterp.h"

using namespace std;

int main(int argc, char* argv[])
{
	int nf1, nf2;
	FLT sigma = 2.0;
	int N1, N2, M;
	if (argc<5) {
		fprintf(stderr,"Usage: ./spreadinterp [method [nupts_distribute [N1 N2 [M [tol]]]]]\n");
		return 1;
	}  
	int method;
	sscanf(argv[1],"%d",&method);

	int nupts_distribute;
	sscanf(argv[2],"%d",&nupts_distribute);

	double w;
	sscanf(argv[3],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[4],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!

	N1 = (int) nf1/sigma;
	N2 = (int) nf2/sigma;
	M = N1*N2;// let density always be 1
	if(argc>5){
		sscanf(argv[5],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
		if(M==0) M=N1*N2;
	}

	FLT tol=1e-6;
	if(argc>6){
		sscanf(argv[6],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}


	int ier;
	int dim=2;
	int ns=std::ceil(-log10(tol/10.0));
	cufinufft_plan dplan;
	FLT upsampfac=2.0;
	cout<<scientific<<setprecision(3);


	FLT *x, *y;
	CPX *c;
	CPX *fws, *fwfinufft;
	cudaMallocHost(&x, M*sizeof(CPX));
	cudaMallocHost(&y, M*sizeof(CPX));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fws,       nf1*nf2*sizeof(CPX));
	cudaMallocHost(&fwfinufft, nf1*nf2*sizeof(CPX));
#if 0
	// spread a single source, only for reference accuracy check...
	c[0].real(1.0); c[0].imag(0.0);   // unit strength
	x[0] = y[0] = nf1/2.0;                  // at center
	ier = cnufftspread(nf1,nf2,1,(FLT*) fwfinufft,1,x,y,NULL,(FLT*) c,opts);
	if (ier!=0) {
		printf("error when spreading M=1 pt for ref acc check (ier=%d)!\n",ier);
		return ier;
	}
	FLT kersumre = 0.0, kersumim = 0.0;  // sum kernel on uniform grid
	for (int i=0;i<nf1*nf2;++i) {
		kersumre += fwfinufft[i].real();
		kersumim += fwfinufft[i].imag();    // in case the kernel isn't real!
	}
#endif
	FLT strre = 0.0, strim = 0.0;          // also sum the strengths
	switch(nupts_distribute){
		// Making data
		case 1: //uniform
			{
				for (int i = 0; i < M; i++) {
					x[i] = RESCALE(M_PI*randm11(), nf1, 1);// x in [-pi,pi)
					y[i] = RESCALE(M_PI*randm11(), nf2, 1);
					c[i].real(randm11());
					c[i].imag(randm11());
					strre += c[i].real();
					strim += c[i].imag();
				}
			}
			break;
		case 2: // concentrate on a small region
			{
				for (int i = 0; i < M; i++) {
					x[i] = RESCALE(M_PI*rand01()/(nf1*2/32), nf1, 1);// x in [-pi,pi)
					y[i] = RESCALE(M_PI*rand01()/(nf1*2/32), nf2, 1);
					c[i].real(randm11());
					c[i].imag(randm11());
					strre += c[i].real();
					strim += c[i].imag();
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

	// Direction 1: Spreading
	printf("[info  ] Type 1: Spreading\n");
#if 0
	FLT sumre = 0.0, sumim = 0.0;   // check spreading accuracy, wrapping
	for (int i=0;i<nf1*nf2;++i) {
		sumre += fwi[i].real();
		sumim += fwi[i].imag();
	}
	FLT pre = kersumre*strre - kersumim*strim;   // pred ans, complex mult
	FLT pim = kersumim*strre + kersumre*strim;
	FLT maxerr = std::max(fabs(sumre-pre), fabs(sumim-pim));
	FLT ansmod = sqrt(sumre*sumre+sumim*sumim);
	printf("    rel err in total over grid:      %.3g\n",maxerr/ansmod);
	// note this is weaker than below dir=2 test, but is good indicator that
	// periodic wrapping is correct
#endif

	/* -------------------------------------- */
	// Method 5: Subprob                     //
	/* -------------------------------------- */
	ier = cufinufft_default_opts(type1, dim, dplan.opts);
	if(ier != 0 ){
		cout<<"error: cufinufft_default_opts"<<endl;
		return 0;
	}
	ier = setup_spreader_for_nufft(dplan.spopts, tol, dplan.opts);

	dplan.opts.upsampfac=upsampfac;
	dplan.opts.gpu_method=method;
	dplan.opts.gpu_kerevalmeth=1;
	dplan.spopts.pirange=0;
	switch(method){
		case 2:
		case 3:
		{
			dplan.opts.gpu_binsizex=32;
			dplan.opts.gpu_binsizey=32;
		}
	}
	timer.restart();
	ier = cufinufft_spread2d(N1, N2, nf1, nf2, fws, M, x, y, c, &dplan);
	FLT tsubprob=timer.elapsedsec();
	if(ier != 0 ){
		cout<<"error: cnufftspread2d_gpu_subprob"<<endl;
		return 0;
	}
	printf("[method %d] %ld NU pts to (%ld,%ld) modes, #%d U pts in %.3g s \t%.3g NU pts/s\n",
			method,M,N1,N2,nf1*nf2,tsubprob,M/tsubprob);

	/* -------------------------------------- */
	// FINUTFFT cpu spreader                  //
	/* -------------------------------------- */
	timer.start();
	spread_opts spopts;
	setup_spreader(spopts,(FLT)tol,upsampfac,1);
	spopts.pirange=0;
	spopts.chkbnds=1;
	spopts.spread_direction=1;
	spopts.flags=0;//ker always return 1
	spopts.kerpad=1;
	spopts.sort_threads=0;
	spopts.sort=2;
	spopts.debug=0;

	ier = spreadinterp(nf1,nf2,1,(FLT*) fwfinufft,M,x,y,NULL,(FLT*) c,spopts);
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
	FLT err=relerrtwonorm(nf1*nf2,fws,fwfinufft);
	err=relerrtwonorm(nf1*nf2,fws,fwfinufft);
	printf("|| fws  - fwfinufft ||_2 / || fws  ||_2 =  %.6g\n", err);

#if 0
	cout<<"[resultdiff]"<<endl;
	FLT fwfinufft_infnorm=infnorm(nf1*nf2, fwfinufft);
	int nn=0;
	for(int j=0; j<nf2; j++){
		for (int i=0; i<nf1; i++){
			if( norm(fwi[i+j*nf1]-fwfinufft[i+j*nf1])/fwfinufft_infnorm > 1e-5 & nn<10){
				cout<<norm(fwi[i+j*nf1]-fwh[i+j*nf1])/fwfinufft_infnorm<<" ";
				cout<<"(i,j)=("<<i<<","<<j<<"), "<<fwi[i+j*nf1] <<","<<fwfinufft[i+j*nf1]<<endl;
				nn++;
			}
		}
	}
	cout<<endl;
#endif
#if 1
	// Direction 2: Interpolation
	printf("\n[info  ] Type 2: Interpolation\n");
	ier = cufinufft_default_opts(type2, dim, dplan.opts);
	ier = setup_spreader_for_nufft(dplan.spopts, tol, dplan.opts);
	dplan.opts.upsampfac=upsampfac;
	dplan.opts.gpu_method=method;
	dplan.opts.gpu_kerevalmeth=1;
	dplan.spopts.pirange=0;
	switch(method){
		case 2:
		case 3:
		{
			dplan.opts.gpu_binsizex=32;
			dplan.opts.gpu_binsizey=32;
		}
	}

	CPX *fw;
	CPX *cfinufft, *cs;
	cudaMallocHost(&fw, nf1*nf2*sizeof(CPX));
	cudaMallocHost(&cfinufft, M*sizeof(CPX));
	cudaMallocHost(&cs,       M*sizeof(CPX));

	for(int i=0; i<nf1*nf2; i++){
	  fw[i].real(1.0);
	  fw[i].imag(0.0);
	}
	/* -------------------------------------- */
	// Method 1: Subprob                      //
	/* -------------------------------------- */
	timer.restart();
	ier = cufinufft_interp2d(N1, N2, nf1, nf2, fw, M, x, y, cs, &dplan);
	FLT tts=timer.elapsedsec();
	if(ier != 0 ){
		cout<<"error: cnufftinterp2d_gpu_subprob"<<endl;
		return 0;
	}
	printf("[method %d] Interp (%ld,%ld) modes to %ld NU pts in %.3g s \t%.3g U pts/s\n",
			  method,nf1,nf2,M,tts,nf1*nf2/tts);
	/* -------------------------------------- */
	// FINUTFFT cpu spreader                  //
	/* -------------------------------------- */
	timer.start();
	setup_spreader(spopts,(FLT)tol,upsampfac,1);
	spopts.pirange=0;
	spopts.chkbnds=1;
	spopts.spread_direction=2;
	spopts.flags=0;//ker always return 1
	spopts.kerpad=1;
	spopts.sort_threads=0;
	spopts.sort=2;
	spopts.debug=0;

	ier = spreadinterp(nf1,nf2,1,(FLT*) fw,M,x,y,NULL,(FLT*) cfinufft,spopts);
	FLT tt=timer.elapsedsec();
	if (ier!=0) {
		printf("error (ier=%d)!\n",ier);
		return ier;
	}
	printf("[finufft] Interp (%ld,%ld) modes to %ld NU pts in %.3g s \t%.3g U pts/s\n",
			  nf1,nf2,M,tt,nf1*nf2/tt);
	err=relerrtwonorm(M,cs,cfinufft);
	printf("|| cs  - cfinufft ||_2 / || cs  ||_2 =  %.6g\n", err);
#if 0
	cout<<"[result-hybrid]"<<endl;
	for(int j=0; j<nf2; j++){
		if( j % opts.gpu_binsizey == 0)
			printf("\n");
		for (int i=0; i<nf1; i++){
			if( i % opts.gpu_binsizex == 0 && i!=0)
				printf(" |");
			printf(" (%2.3g,%2.3g)",fwi[i+j*nf1].real(),fwi[i+j*nf1].imag() );
			//cout<<" "<<setw(8)<<fwfinufft[i+j*nf1];
		}
		cout<<endl;
	}
	cout<<endl;
#endif
#endif
	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fws);
	cudaFreeHost(fwfinufft);
	return 0;
}
