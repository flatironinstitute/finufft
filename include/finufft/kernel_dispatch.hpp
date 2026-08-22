// The kernel width, the Horner degree and the dimension are runtime values that the
// spread and interp code takes as template parameters. Both directions turn them into
// template arguments here.

#pragma once

#include <finufft_common/constants.h>

#include <poet/poet.hpp> // poet::dispatch / inclusive_range / dispatch_param

#include <tuple>
#include <type_traits>

namespace finufft::spreadinterp {

// Calls caller.operator()<nspread, nc>(). caller reports an unsupported pair itself.
template<typename TF, typename Caller>
void kernel_dispatch(int nspread, int nc, Caller caller) noexcept {
  using namespace finufft::common;
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<TF>>;
  using NcSeq = poet::inclusive_range<MIN_NC, MAX_NC>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{nspread},
                                         poet::dispatch_param<NcSeq>{nc}));
}

// Calls f(std::integral_constant<int, ndims>{}) for ndims 1, 2 or 3.
template<typename F> void dispatch_ndims(int ndims, F &&f) {
  if (ndims == 1)
    f(std::integral_constant<int, 1>{});
  else if (ndims == 2)
    f(std::integral_constant<int, 2>{});
  else
    f(std::integral_constant<int, 3>{});
}

} // namespace finufft::spreadinterp
