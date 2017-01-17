#include "../src/cnufftspread.h"

#include <vector>
#include <math.h>

double rand01() {
    return (rand()%RAND_MAX)*1.0/RAND_MAX;
}

int main(int argc, char* argv[])
/* This is the test code for the 3D C++ spreader, both directions.
 * It checks speed and basic correctness via the grid sum of the result.
 *
 * Magland and Barnett 1/14/17
 */
{
  long M=1e6;    // choose problem size:  # NU pts
  long N=100;    //                       Fourier grid size
  std::vector<double> kx(M),ky(M),kz(M),d_nonuniform(2*M);    // Re & Im
  std::vector<double> d_uniform(N*N*N*2);              // N^3 for Re and Im

  cnufftspread_opts opts; // set method opts...
  opts.debug = 0;
  opts.sort_data=true;    // 50% faster on i7
  double tol = 1e-6;        // choose tol (1e6 has nspread=8)
  set_kb_opts_from_eps(opts,tol);

    // test direction 1 (NU -> U spreading) ..............................
    opts.spread_direction=1;
    printf("cnufftspread 3D, dir=%d, tol=%.3g: nspread=%d\n",opts.spread_direction,tol,opts.nspread);

    // spread a single source for reference...
    d_nonuniform[0] = 1.0; d_nonuniform[1] = 0.0;   // unit strength
    kx[0] = ky[0] = kz[0] = 0.0;                    // at center
    int ier = cnufftspread(N,N,N,d_uniform.data(),1,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    double kersumre = 0.0, kersumim = 0.0;  // sum kernel on uniform grid
    for (long i=0;i<N*N*N;++i) {
      kersumre += d_uniform[2*i]; 
      kersumim += d_uniform[2*i+1];    // in case the kernel isn't real!
    }

    // now do the large-scale test w/ random sources..
    srand(0);    // fix seed for reproducibility
    double strre = 0.0, strim = 0.0;          // also sum the strengths
    for (long i=0; i<M; ++i) {
        kx[i]=rand01()*N;
        ky[i]=rand01()*N;
        kz[i]=rand01()*N;
        d_nonuniform[i*2]=rand01()*2-1;
        d_nonuniform[i*2+1]=rand01()*2-1;
	strre += d_nonuniform[2*i]; 
	strim += d_nonuniform[2*i+1];
    }
    CNTime timer; timer.start();
    ier = cnufftspread(N,N,N,d_uniform.data(),M,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    double t=timer.elapsedsec();
    printf("(ier=%d)\t%ld pts in %.3g s \t%.3g NU pts/s \t%.3g spread pts/s\n",ier,M,t,M/t,pow(opts.nspread,3)*M/t);

    double sumre = 0.0, sumim = 0.0;   // check spreading accuracy, wrapping
    for (long i=0;i<N*N*N;++i) {
      sumre += d_uniform[2*i]; 
      sumim += d_uniform[2*i+1];
    }
    double pre = kersumre*strre - kersumim*strim;   // pred ans, complex mult
    double pim = kersumim*strre + kersumre*strim;
    double maxerr = std::max(sumre-pre, sumim-pim);
    double ansmod = sqrt(sumre*sumre+sumim*sumim);
    printf("\trel err in total on grid: %.3g\n",maxerr/ansmod);
    // note this cannot be correct unless periodic wrapping is correct


    // test direction 2 (U -> NU interpolation) ..............................
    opts.spread_direction=2;
    printf("cnufftspread 3D, dir=%d, tol=%.3g: nspread=%d\n",opts.spread_direction,tol,opts.nspread);

    for (long i=0;i<N*N*N;++i) {     // unit grid data
      d_uniform[2*i] = 1.0;
      d_uniform[2*i+1] = 0.0;
    }
    for (long i=0; i<M; ++i) {       // random target pts
        kx[i]=rand01()*N;
        ky[i]=rand01()*N;
        kz[i]=rand01()*N;
    }
    timer.restart();
    ier = cnufftspread(N,N,N,d_uniform.data(),M,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    t=timer.elapsedsec();
    printf("(ier=%d)\t%ld pts in %.3g s \t%.3g NU pts/s \t%.3g spread pts/s\n",ier,M,t,M/t,pow(opts.nspread,3)*M/t);

    // math test is worst-case error from pred value (kersum) on interp pts:
    maxerr = 0.0;
    for (long i=0;i<M;++i) {
      double err = std::max(fabs(d_nonuniform[2*i]-kersumre),
			    fabs(d_nonuniform[2*i+1]-kersumim));
      if (err>maxerr) maxerr=err;
    }
    ansmod = sqrt(kersumre*kersumre+kersumim*kersumim);
    printf("\tmax rel err in values at NU pts: %.3g\n",maxerr/ansmod);
    // this is weaker test than for dir=1, since it cannot detect reading
    // from wrong grid pts (they are all unity)

    return 0;
}
