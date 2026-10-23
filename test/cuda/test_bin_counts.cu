#include <climits>
#include <cstdint>

#include <cufinufft/spreadinterp.hpp>

// assert.h goes last: a later include re-evaluates NDEBUG and turns the
// asserts back into no-ops in a Release build
#undef NDEBUG
#include <assert.h>

// Host asserts for the integer bin and subprob counts (issue #890).
int main() {
  using cufinufft::spreadinterp::ceil_div;
  using cufinufft::spreadinterp::get_nbins;
  using cufinufft::spreadinterp::nbins_total;

  // runtime inputs: a constant folds in exact precision and hides the int
  // overflow that these count forms must avoid
  volatile int imax  = INT_MAX;
  volatile int bs    = 1024;
  volatile int zero  = 0;
  volatile int n2048 = 2048;

  // ceil_div at the boundary where the +b-1 sum overflows int
  assert(ceil_div(imax, bs) == 2097152);
  assert(ceil_div(imax - 500, bs) == 2097152);
  // the ceil must keep 0 subprobs at 0 points
  assert(ceil_div(zero, bs) == 0);
  // get_nbins at INT_MAX: the nf + binsize - 1 form overflows int
  assert(get_nbins<1>({imax, 1, 1}, {bs, 1, 1})[0] == 2097152);
  // nbins_total product in int64: the int product wraps at 2048^3 bins
  assert(nbins_total({n2048, n2048, n2048}) == 8589934592LL);
  return 0;
}
