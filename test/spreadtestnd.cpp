#include "../src/cnufftspread.h"
#include <vector>
#include <math.h>
#include <stdio.h>

int usage()
{
  printf("usage: spreadtestnd [dim [M [N [tol [sort]]]]]\n\twhere dim=1,2 or 3\n\tM=# nonuniform pts\n\tN=# uniform pts\n\ttol=requested accuracy\n\tsort=0 (don't sort data) or 1 (do, default)\n");
}

int main(int argc, char* argv[])
/* Test executable for the 1D, 2D, or 3D C++ spreader, both directions.
 * It checks speed, and basic correctness via the grid sum of the result.
 * Usage: spreadtestnd [dim [M [N [tol [sort]]]]]
 *	  where dim = 1,2 or 3
 *	  M = # nonuniform pts
 *        N = # uniform pts
 *	  tol = requested accuracy
 *        sort = 0 (don't sort data) or 1 (do, default)
 *
 * Example: spreadtestnd 3 1e7 1e6 1e-6 0
 *
 * Note: for 2d and 3d, sort=1 is 1-2x faster on i7; but 0.5-0.9x (ie slower) on xeon!
 *
 * Compilation (also check ../makefile):
 *    g++ spreadtestnd.cpp ../src/cnufftspread.o ../src/utils.o -o spreadtestnd -fPIC -Ofast -funroll-loops -std=c++11 -fopenmp
 *
 * Magland, expanded by Barnett 1/14/17. Better cmd line args 3/13/17
 * indep setting N 3/27/17. parallel rand() & sort flag 3/28/17
 */
{
  int d = 3;            // default #dims
  double tol = 1e-6;    // default (eg 1e-6 has nspread=7)
  BIGINT M = 1e6;         // default # NU pts
  BIGINT roughNg = 1e6;   // default # U pts
  int sort = 1;         // default
  if (argc<=1) { usage(); return 0; }
  sscanf(argv[1],"%d",&d);
  if (d<1 || d>3) {
    printf("d must be 1, 2 or 3!\n"); usage(); return 1;
  }
  if (argc>2) {
    double w; sscanf(argv[2],"%lf",&w); M = (BIGINT)w;  // so can read 1e6 right!
    if (M<1) {
      printf("M (# NU pts) must be positive!\n"); usage(); return 1;
    }
  }
  if (argc>2) {
    double w; sscanf(argv[3],"%lf",&w); roughNg = (BIGINT)w;
    if (roughNg<1) {
      printf("N (# U pts) must be positive!\n"); usage(); return 1;
    }
  }
  if (argc>4) {
    sscanf(argv[4],"%lf",&tol);
    if (tol<=0.0) {
      printf("tol must be positive!\n"); usage(); return 1;
    }
  }
  if (argc>5) {
    sscanf(argv[5],"%d",&sort);
    if (sort!=0 && sort!=1) {
      printf("sort must be 0 or 1!\n"); usage(); return 1;
    }
  }
  if (argc>6) { usage();
    return 1; }
  BIGINT N=std::round(pow(roughNg,1.0/d));         // Fourier grid size per dim
  BIGINT Ng = (BIGINT)pow(N,d);                      // actual total grid points
  BIGINT N2 = (d>=2) ? N : 1, N3 = (d==3) ? N : 1;    // the y and z grid sizes
  std::vector<FLT> kx(M),ky(1),kz(1),d_nonuniform(2*M);    // NU, Re & Im
  if (d>1) ky.resize(M);                           // only alloc needed coords
  if (d>2) kz.resize(M);
  std::vector<FLT> d_uniform(2*Ng);                        // Re and Im

  spread_opts opts; // set method opts...
  opts.debug = 0;
  opts.sort_data=(bool)sort;   // for 3D: 1-2x faster on i7; but 0.5-0.9x (ie slower) on xeon!
  FLT Rdummy = 2.0;    // since no nufft done, merely to please the setup
  setup_kernel(opts,(FLT)tol,Rdummy);  // note tol is always double

    // test direction 1 (NU -> U spreading) ..............................
    opts.spread_direction=1;
    printf("cnufftspread %dD, %.3g U pts, dir=%d, tol=%.3g: nspread=%d\n",d,(double)Ng,opts.spread_direction,tol,opts.nspread);

    // spread a single source for reference...
    d_nonuniform[0] = 1.0; d_nonuniform[1] = 0.0;   // unit strength
    kx[0] = ky[0] = kz[0] = N/2;                    // at center
    int ier = cnufftspread(N,N2,N3,d_uniform.data(),1,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    FLT kersumre = 0.0, kersumim = 0.0;  // sum kernel on uniform grid
    for (BIGINT i=0;i<Ng;++i) {
      kersumre += d_uniform[2*i]; 
      kersumim += d_uniform[2*i+1];    // in case the kernel isn't real!
    }

    // now do the large-scale test w/ random sources..
    printf("making random data...\n");
    FLT strre = 0.0, strim = 0.0;          // also sum the strengths
#pragma omp parallel
    {
      unsigned int se=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,1000000) reduction(+:strre,strim)
    for (BIGINT i=0; i<M; ++i) {
      kx[i]=rand01r(&se)*N;
      if (d>1) ky[i]=rand01r(&se)*N;              // only fill needed coords
      if (d>2) kz[i]=rand01r(&se)*N;
      d_nonuniform[i*2]=randm11r(&se);
      d_nonuniform[i*2+1]=randm11r(&se);
      strre += d_nonuniform[2*i]; 
      strim += d_nonuniform[2*i+1];
    }
    }
    printf("calling spreader...\n");
    CNTime timer; timer.start();
    ier = cnufftspread(N,N2,N3,d_uniform.data(),M,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    double t=timer.elapsedsec();
    if (ier!=0) {
      printf("error (ier=%d)!\n",ier);
      return ier;
    } else
      printf("\t%.3g NU pts in %.3g s \t%.3g pts/s \t%.3g spread pts/s\n",(double)M,t,M/t,pow(opts.nspread,d)*M/t);

    FLT sumre = 0.0, sumim = 0.0;   // check spreading accuracy, wrapping
#pragma omp parallel for reduction(+:sumre,sumim)
    for (BIGINT i=0;i<Ng;++i) {
      sumre += d_uniform[2*i]; 
      sumim += d_uniform[2*i+1];
    }
    FLT pre = kersumre*strre - kersumim*strim;   // pred ans, complex mult
    FLT pim = kersumim*strre + kersumre*strim;
    FLT maxerr = std::max(sumre-pre, sumim-pim);
    FLT ansmod = sqrt(sumre*sumre+sumim*sumim);
    printf("\trel err in total over grid:      %.3g\n",maxerr/ansmod);
    // note this is weaker than below dir=2 test, but is good indicator that
    // periodic wrapping is correct


    // test direction 2 (U -> NU interpolation) ..............................
    opts.spread_direction=2;
    printf("cnufftspread %dD, %.3g U pts, dir=%d, tol=%.3g: nspread=%d\n",d,(double)Ng,opts.spread_direction,tol,opts.nspread);

    printf("making random data...\n");
    for (BIGINT i=0;i<Ng;++i) {     // unit grid data
      d_uniform[2*i] = 1.0;
      d_uniform[2*i+1] = 0.0;
    }
#pragma omp parallel
    {
      unsigned int s=MY_OMP_GET_THREAD_NUM();  // needed for parallel random #s
#pragma omp for schedule(dynamic,1000000)
      for (BIGINT i=0; i<M; ++i) {       // random target pts
        kx[i]=rand01r(&s)*N;
	if (d>1) ky[i]=rand01r(&s)*N;
	if (d>2) kz[i]=rand01r(&s)*N;
      }
    }
    printf("calling spreader...\n");
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
    for (BIGINT i=0;i<M;++i) {
      FLT err = std::max(fabs(d_nonuniform[2*i]-kersumre),
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
