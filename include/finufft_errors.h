#ifndef FINUFFT_ERRORS_H
#define FINUFFT_ERRORS_H

// ---------- Global error/warning output codes for the library ---------------
// NB: if change these numbers, also must regen test/results/dumbinputs.refout
#define FINUFFT_WARN_EPS_TOO_SMALL         1
// this means that a fine grid array dim exceeded MAX_NF; no malloc tried...
#define FINUFFT_ERR_MAXNALLOC              2
#define FINUFFT_ERR_SPREAD_BOX_SMALL       3
#define FINUFFT_ERR_SPREAD_PTS_OUT_RANGE   4
#define FINUFFT_ERR_SPREAD_ALLOC           5
#define FINUFFT_ERR_SPREAD_DIR             6
#define FINUFFT_ERR_UPSAMPFAC_TOO_SMALL    7
#define FINUFFT_ERR_HORNER_WRONG_BETA      8
#define FINUFFT_ERR_NTRANS_NOTVALID        9
#define FINUFFT_ERR_TYPE_NOTVALID          10
// some generic internal allocation failure...
#define FINUFFT_ERR_ALLOC                  11
#define FINUFFT_ERR_DIM_NOTVALID           12
#define FINUFFT_ERR_SPREAD_THREAD_NOTVALID 13
#define FINUFFT_ERR_NDATA_NOTVALID         14
// cuda malloc/memset/kernel failure/etc
#define FINUFFT_ERR_CUDA_FAILURE           15
#define FINUFFT_ERR_PLAN_NOTVALID          16
#define FINUFFT_ERR_METHOD_NOTVALID        17
#define FINUFFT_ERR_BINSIZE_NOTVALID       18
#define FINUFFT_ERR_INSUFFICIENT_SHMEM     19

#endif
