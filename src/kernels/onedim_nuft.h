/** Inner loops for type 3 1-d kernel functionality.
 *
 */

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace finufft {
// Plain C++ scalar version of the inner loop.
template <typename FT>
void onedim_nuft_kernel_scalar(
    size_t nk, int q, FT const *f, FT const *z, FT const* k, FT *phihat) {
    for (size_t j = 0; j < nk; ++j) { // loop along output array
        FT x = 0.0;                   // register
        for (int n = 0; n < q; ++n)
            x += f[n] * 2 * std::cos(k[j] * z[n]); // pos & neg freq pair.  use FLT cos!
        phihat[j] = x;
    }
}
} // namespace finufft