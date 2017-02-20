#include "../src/cnufftspread.h"
#include "../src/utils.h"
#include <stdio.h>


// some cheb eval test data for local speed test only
static double A[] = {
  -4.41534164647933937950E-18,
  3.33079451882223809783E-17,
  -2.43127984654795469359E-16,
  1.71539128555513303061E-15,
  -1.16853328779934516808E-14,
  7.67618549860493561688E-14,
  -4.85644678311192946090E-13,
  2.95505266312963983461E-12,
  -1.72682629144155570723E-11,
  9.67580903537323691224E-11,
  -5.18979560163526290666E-10,
  2.65982372468238665035E-9,
  -1.30002500998624804212E-8,
  6.04699502254191894932E-8,
  -2.67079385394061173391E-7,
  1.11738753912010371815E-6,
  -4.41673835845875056359E-6,
  1.64484480707288970893E-5,
  -5.75419501008210370398E-5,
  1.88502885095841655729E-4,
  -5.76375574538582365885E-4,
  1.63947561694133579842E-3,
  -4.32430999505057594430E-3,
  1.05464603945949983183E-2,
  -2.37374148058994688156E-2,
  4.93052842396707084878E-2,
  -9.49010970480476444210E-2,
  1.71620901522208775349E-1,
  -3.04682672343198398683E-1,
  6.76795274409476084995E-1
};

double chbevl_local(double x, double array[], int n)  // speed test
{
  double b0, b1, b2, *p;
  int i;

  p = array;
  b0 = *p++;
  b1 = 0.0;
  i = n - 1;

  do {
	b2 = b1;
	b1 = b0;
	b0 = x * b1  -  b2  + *p++;
  }
  while (--i);

  return(0.5*(b0 - b2));
}

double polevl(double x,double array[], int n)
{
  double y=0.0;
  double x2 = x*x;
  //if (((int)(1e8*x2) % 2) > 0)       // see if conditionals slow it down
    for (int i=0;i<n;++i) y = (y + array[i])*x2;
  //else
  //  for (int i=0;i<n;++i) y = (y + array[i+4])*x2;
  return y;
}


int main(int argc, char* argv[])
// Output kernels for various precisions. Each row is z, phi_1(z), phi_2(z),...
// where phi_j are the kernels for various precisions.
// Also times the kernel eval and various styles of approximation for it.
// Barnett 2/10/17
{
  const double tols[] = {1e-1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12,1e-14};
  const int ntols = 8;
  spread_opts spopts;

  int N=1e4;       // plot pts
  double z0 = 10;   // plot range
  for (int n=0;n<N;++n) {
    double z = z0*(2*n/(double)N - 1.0);  // ordinate
    printf("%.15g\t",z);
    for (int t=0;t<ntols;++t) {
      //      int ier_set = set_KB_opts_from_eps(spopts,tols[t]);
      int ier_set = setup_kernel(spopts,tols[t],2.0);
      printf("%.15g\t",evaluate_kernel(z,spopts));
    }
    printf("\n");
  }
  // done w/ stdout for plot.


  // bunch of timing expts, report to stderr...

  int ier_set = setup_kernel(spopts,1e-4,2.0);
  N=1e7;  // how many evals to test
  z0 = spopts.nspread/2 - 0.5;   // half-width
  double dz=2*z0/(double)N;
  double y = 0.0;   // running total to prevent -O3 removing the needed calcs!
  CNTime timer; timer.start();
  for (int n=0;n<N;++n) {
    double z = -z0 + n*dz;      // ordinate, something easy to calc
    y += evaluate_kernel(z,spopts);
  }
  fprintf(stderr,"%.3g ns per kernel eval \t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);

  timer.restart(); y = 0.0;
  for (int n=0;n<N;++n) {
    double z = -z0 + n*dz;
    y += exp(z);
  }
  fprintf(stderr,"%.3g ns per real exp() \t\t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);

  timer.restart(); y = 0.0;
  for (int n=0;n<N;++n) {
    double z = -z0 + n*dz;
    y += z;   // (int)(z*1e9) % 1000;   // mod seems only 1.5 ns - why?
  }
  fprintf(stderr,"%.3g ns per NOP \t\t\t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);

  dz = 2/(double)N;
  timer.restart();  y = 0.0;
  for (int n=0;n<N;++n) {
    double z = -1 + n*dz;
    y += chbevl_local(z,A,30);
  }
  fprintf(stderr,"%.3g ns per 30-term cheb \t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);

  dz = 2/(double)N;
  timer.restart();  y = 0.0;
  for (int n=0;n<N;++n) {
    double z = -1 + n*dz;
    double u = chbevl_local(z,A,16);
    if (u-(int)u < 0.5)              // too easy to predict
      y += u;
    else
      y += 100.0;
  }
  fprintf(stderr,"%.3g ns per 16-term cheb w/ if \t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);

    timer.restart();  y = 0.0;
  for (int n=0;n<N;++n) {
    double z = -1 + n*dz;
    y += exp(chbevl_local(z,A,4));
  }
  fprintf(stderr,"%.3g ns per exp(4-term cheb) \t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);
  
    timer.restart();  y = 0.0;
  for (int n=0;n<N;++n) {
    double z = -1 + n*dz;
    y += exp(polevl(asin(z),A,4));
    //double q = asin(z);
    //y += exp(0.7 + (1.2 + (2.4 + (2.5 + 3.1*q)*q)*q)*q);
  }
  fprintf(stderr,"%.3g ns per exp(4-term poly(asin()) \t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);
  
    timer.restart();  y = 0.0;
  for (int n=0;n<N;++n) {
    double z = -1 + n*dz;
    if (abs(z)<0.99999) {
      double q = sqrt(1-z*z);
      y += exp(56.1*q)/sqrt(q);
    }
  }
  fprintf(stderr,"%.3g ns per exp(sqrt)/sqrt(sqrt) \t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);

     timer.restart();  y = 0.0;
  for (int n=0;n<N;++n) {
    double z = -1 + n*dz;
    if (abs(z)<1) {
      double q = sqrt(1-z*z);
      y += exp(56.1*q); ///sqrt(q);
    }
  }
  fprintf(stderr,"%.3g ns per exp(sqrt) \t(dummy y=%.3g)\n",timer.elapsedsec()/N*1e9,y);
 
  return 0;
}
