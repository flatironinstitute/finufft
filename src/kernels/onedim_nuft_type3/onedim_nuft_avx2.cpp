#include "onedim_nuft.h"
#include "onedim_nuft_impl.h"

namespace finufft {

INSTANTIATE_NUFT_IMPLEMENTATIONS_WITH_WIDTH(onedim_nuft_kernel_avx2, 8);

}
