#include "../src/cnufftspread.h"
#include <vector>
#include <math.h>
#include <stdio.h>

int usage()
{
  printf("usage: spreadtestnd [dim [M [tol]]]\n\twhere dim=1,2 or 3\n\tM=problem size, sets both # NU and U pts\n\ttol=requested accuracy\n");
}

int main(int argc, char* argv[])
/* Test executable for the 1D, 2D, or 3D C++ spreader, both directions.
 * It checks speed, and basic correctness via the grid sum of the result.
 * Usage: spreadtestnd [dim [M [tol]]]
 *	  where dim=1,2 or 3
 *	  M=problem size, sets both # NU and U pts
 *	  tol=requested accuracy
 *
 * Example: spreadtestnd 3 1e6 1e-6
 *
 * Compilation (also check ../makefile):
 *    g++ spreadtestnd.cpp ../src/cnufftspread.o ../src/utils.o -o spreadtestnd -fPIC -Ofast -funroll-loops -march=native -std=c++11 -fopenmp
 *
 * Magland, expanded by Barnett 1/14/17. Better cmd line args 3/13/17
 */
{
  int d = 3;            // default
  double tol = 1e-6;    // default (1e6 has nspread=8)
  long M = 1e6;         // default problem size (# NU pts)
  if (argc<=1) { usage(); return 0; }
  sscanf(argv[1],"%d",&d);
  if (d<1 || d>3) {
    printf("d must be 1, 2 or 3!\n"); usage(); return 1;
  }
  if (argc>2) {
    double w; sscanf(argv[2],"%lf",&w); M = (long)w;  // so can read 1e6 right!
    if (M<1) {
      printf("M (problem size) must be positive!\n"); usage(); return 1;
    }
  }
  if (argc>3) {
    sscanf(argv[3],"%lf",&tol);
    if (tol<=0.0) {
      printf("tol must be positive!\n"); usage(); return 1;
    }
  }
  if (argc>4) { usage();
    return 1; }
  long roughNg = M;                          // # grid pts = # NU pts
  long N=(long)(pow(roughNg,1.0/d));         // Fourier grid size per dim
  long Ng = (long)pow(N,d);                  // actual total grid points
  long N2 = (d>=2) ? N : 1, N3 = (d==3) ? N : 1;    // the y and z grid sizes
  std::vector<double> kx(M),ky(M),kz(M),d_nonuniform(2*M);    // NU, Re & Im
  std::vector<double> d_uniform(2*Ng);                        // Re and Im

  spread_opts opts; // set method opts...
  opts.debug = 0;
  opts.sort_data=true;    // 50% faster on i7
  double Rdummy = 2.0;    // since no nufft done, this is to please the setup
  setup_kernel(opts,tol,Rdummy);

    // test direction 1 (NU -> U spreading) ..............................
    opts.spread_direction=1;
    printf("cnufftspread %dD, %.3g U pts, dir=%d, tol=%.3g: nspread=%d\n",d,(double)Ng,opts.spread_direction,tol,opts.nspread);

    // spread a single source for reference...
    d_nonuniform[0] = 1.0; d_nonuniform[1] = 0.0;   // unit strength
    kx[0] = ky[0] = kz[0] = N/2;                    // at center
    int ier = cnufftspread(N,N2,N3,d_uniform.data(),1,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    double kersumre = 0.0, kersumim = 0.0;  // sum kernel on uniform grid
    for (long i=0;i<Ng;++i) {
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
      d_nonuniform[i*2]=randm11();
      d_nonuniform[i*2+1]=randm11();
      strre += d_nonuniform[2*i]; 
      strim += d_nonuniform[2*i+1];
    }
    CNTime timer; timer.start();
    ier = cnufftspread(N,N2,N3,d_uniform.data(),M,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    double t=timer.elapsedsec();
    if (ier!=0) {
      printf("error (ier=%d)!\n",ier);
      return 1;
    } else
      printf("\t%.3g NU pts in %.3g s \t%.3g pts/s \t%.3g spread pts/s\n",(double)M,t,M/t,pow(opts.nspread,d)*M/t);

    double sumre = 0.0, sumim = 0.0;   // check spreading accuracy, wrapping
    for (long i=0;i<Ng;++i) {
      sumre += d_uniform[2*i]; 
      sumim += d_uniform[2*i+1];
    }
    double pre = kersumre*strre - kersumim*strim;   // pred ans, complex mult
    double pim = kersumim*strre + kersumre*strim;
    double maxerr = std::max(sumre-pre, sumim-pim);
    double ansmod = sqrt(sumre*sumre+sumim*sumim);
    printf("\trel err in total over grid:      %.3g\n",maxerr/ansmod);
    // note this is weaker than below dir=2 test, but is good indicator that
    // periodic wrapping is correct


    // test direction 2 (U -> NU interpolation) ..............................
    opts.spread_direction=2;
    printf("cnufftspread %dD, %.3g U pts, dir=%d, tol=%.3g: nspread=%d\n",d,(double)Ng,opts.spread_direction,tol,opts.nspread);

    for (long i=0;i<Ng;++i) {     // unit grid data
      d_uniform[2*i] = 1.0;
      d_uniform[2*i+1] = 0.0;
    }
    for (long i=0; i<M; ++i) {       // random target pts
      kx[i]=rand01()*N; //***
      ky[i]=rand01()*N;
      kz[i]=rand01()*N;
    }
    timer.restart();
    ier = cnufftspread(N,N2,N3,d_uniform.data(),M,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    t=timer.elapsedsec();
    if (ier!=0) {
      printf("error (ier=%d)!\n",ier);
      return 1;
    } else
      printf("\t%.3g NU pts in %.3g s \t%.3g pts/s \t%.3g spread pts/s\n",(double)M,t,M/t,pow(opts.nspread,d)*M/t);

    // math test is worst-case error from pred value (kersum) on interp pts:
    maxerr = 0.0;
    for (long i=0;i<M;++i) {
      double err = std::max(fabs(d_nonuniform[2*i]-kersumre),
			    fabs(d_nonuniform[2*i+1]-kersumim));
      if (err>maxerr) maxerr=err;
    }
    ansmod = sqrt(kersumre*kersumre+kersumim*kersumim);
    printf("\tmax rel err in values at NU pts: %.3g\n",maxerr/ansmod);
    // this is stronger test than for dir=1, since it tests sum of kernel for
    // each NU pt. However, it cannot detect reading
    // from wrong grid pts (they are all unity)

    return 0;
}
