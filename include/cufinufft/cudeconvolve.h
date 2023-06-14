#ifndef __CUDECONVOLVE_H__
#define __CUDECONVOLVE_H__

#include <cufinufft_eitherprec.h>

namespace cufinufft {
namespace deconvolve {
__global__ void deconvolve_1d(int ms, int nf1, int fw_width, CUCPX *fw, CUCPX *fk, CUFINUFFT_FLT *fwkerhalf1);
__global__ void amplify_1d(int ms, int nf1, int fw_width, CUCPX *fw, CUCPX *fk, CUFINUFFT_FLT *fwkerhalf2);

__global__ void deconvolve_2d(int ms, int mt, int nf1, int nf2, int fw_width, CUCPX *fw, CUCPX *fk,
                              CUFINUFFT_FLT *fwkerhalf1, CUFINUFFT_FLT *fwkerhalf2);
__global__ void amplify_2d(int ms, int mt, int nf1, int nf2, int fw_width, CUCPX *fw, CUCPX *fk,
                           CUFINUFFT_FLT *fwkerhalf1, CUFINUFFT_FLT *fwkerhalf2);

__global__ void deconvolve_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, int fw_width, CUCPX *fw, CUCPX *fk,
                              CUFINUFFT_FLT *fwkerhalf1, CUFINUFFT_FLT *fwkerhalf2, CUFINUFFT_FLT *fwkerhalf3);
__global__ void amplify_3d(int ms, int mt, int mu, int nf1, int nf2, int nf3, int fw_width, CUCPX *fw, CUCPX *fk,
                           CUFINUFFT_FLT *fwkerhalf1, CUFINUFFT_FLT *fwkerhalf2, CUFINUFFT_FLT *fwkerhalf3);

int CUDECONVOLVE1D(CUFINUFFT_PLAN d_mem, int blksize);
int CUDECONVOLVE2D(CUFINUFFT_PLAN d_mem, int blksize);
int CUDECONVOLVE3D(CUFINUFFT_PLAN d_mem, int blksize);
} // namespace deconvolve
} // namespace cufinufft
#endif
