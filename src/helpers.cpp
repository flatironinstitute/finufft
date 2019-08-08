#include <helpers.h>
#include <nufft_opts.h>
#include <fftw_defs.h>
#include <finufft.h>

extern "C" void finufft_default_opts(nufft_opts *o)
// Sets default nufft opts. See finufft.h for definition of opts.
// This was created to avoid uncertainty about C++11 style static initialization
// when called from MEX. Barnett 10/30/17
{
  o->upsampfac = 2.0f;   // sigma: either 2.0, or 1.25 for smaller RAM, FFTs
  o->chkbnds = 0;
  o->debug = 0;
  o->spread_debug = 0;
  o->spread_sort = 2;        // use heuristic rule for whether to sort
  o->spread_kerevalmeth = 1; // 0: direct exp(sqrt()), 1: Horner ppval
  o->spread_kerpad = 1;      // (relevant iff kerevalmeth=0)
  o->fftw = FFTW_ESTIMATE;   // use FFTW_MEASURE for slow first call, fast rerun
  o->modeord = 0;
}



BIGINT next235even(BIGINT n)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth"). Adapted from fortran in hellskitchen.  Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
{
  if (n<=2) return 2;
  if (n%2 == 1) n+=1;   // even
  BIGINT nplus = n-2;   // to cancel out the +=2 at start of loop
  BIGINT numdiv = 2;    // a dummy that is >1
  while (numdiv>1) {
    nplus += 2;         // stays even
    numdiv = nplus;
    while (numdiv%2 == 0) numdiv /= 2;  // remove all factors of 2,3,5...
    while (numdiv%3 == 0) numdiv /= 3;
    while (numdiv%5 == 0) numdiv /= 5;
  }
  return nplus;
}

// ----------------------- helpers for timing (always stay double prec)...
using namespace std;

void CNTime::start()
{
  gettimeofday(&initial, 0);
}

double CNTime::restart()
// Barnett changed to returning in sec
{
  double delta = this->elapsedsec();
  this->start();
  return delta;
}

double CNTime::elapsedsec()
// returns answers as double, in seconds, to microsec accuracy. Barnett 5/22/18
{
  struct timeval now;
  gettimeofday(&now, 0);
  double nowsec = (double)now.tv_sec + 1e-6*now.tv_usec;
  double initialsec = (double)initial.tv_sec + 1e-6*initial.tv_usec;
  return nowsec - initialsec;
}

int ndims_from_Ns(BIGINT N1, BIGINT N2, BIGINT N3)
/* rule for getting number of spreading dimensions from the list of Ns per dim.
   Split out, Barnett 7/26/18
*/
{
  int ndims = 1;                // decide ndims: 1,2 or 3
  if (N2>1) ++ndims;
  if (N3>1) ++ndims;
  return ndims;
}
