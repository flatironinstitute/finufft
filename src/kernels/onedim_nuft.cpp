#include "onedim_nuft.h"

#include <xmmintrin.h>
#include <pmmintrin.h>

namespace finufft {

DisableDenormals::DisableDenormals() noexcept
    : ftz_mode(_MM_GET_FLUSH_ZERO_MODE()),
      daz_mode(_MM_GET_DENORMALS_ZERO_MODE()) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

DisableDenormals::~DisableDenormals() noexcept {
    _MM_SET_FLUSH_ZERO_MODE(ftz_mode);
    _MM_SET_DENORMALS_ZERO_MODE(daz_mode);
}

}
