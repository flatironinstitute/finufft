#include <cufinufft/utils.h>

namespace cufinufft {
namespace utils {
CUFINUFFT_BIGINT next235beven(CUFINUFFT_BIGINT n, CUFINUFFT_BIGINT b)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth") and is a multiple of b (b is a number that the only prime
// factors are 2,3,5). Adapted from fortran in hellskitchen. Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
// added condition about b Melody 05/31/20
{
  if (n <= 2) return 2;
  if (n % 2 == 1) n += 1;                // even
  CUFINUFFT_BIGINT nplus  = n - 2;       // to cancel out the +=2 at start of loop
  CUFINUFFT_BIGINT numdiv = 2;           // a dummy that is >1
  while ((numdiv > 1) || (nplus % b != 0)) {
    nplus += 2;                          // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0) numdiv /= 2; // remove all factors of 2,3,5...
    while (numdiv % 3 == 0) numdiv /= 3;
    while (numdiv % 5 == 0) numdiv /= 5;
  }
  return nplus;
}

// ----------------------- helpers for timing (always stay double prec)...

void CNTime::start() { gettimeofday(&initial, 0); }

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
  double nowsec     = (double)now.tv_sec + 1e-6 * now.tv_usec;
  double initialsec = (double)initial.tv_sec + 1e-6 * initial.tv_usec;
  return nowsec - initialsec;
}

} // namespace utils
} // namespace cufinufft
