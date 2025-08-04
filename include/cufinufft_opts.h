#ifndef __CUFINUFFT_OPTS_H__
#define __CUFINUFFT_OPTS_H__

// python/cufinufft/cufinufft/_cufinufft.py and matlab/cufinufft.mw
// must be kept in sync with this struct

typedef struct cufinufft_opts { // see cufinufft_default_opts() for defaults
  double upsampfac; // upsampling ratio sigma, only 2.0 (standard) is implemented
                    /* following options are for gpu */
  int gpu_method;   // 1: nonuniform-pts driven, 2: shared mem (SM), 3: output driven (OD)
  int gpu_sort;     // when NU-pts driven: 0: no sort (GM), 1: sort (GM-sort)

  int gpu_binsizex; // used for 2D, 3D subproblem method and Output Driven
  int gpu_binsizey;
  int gpu_binsizez;

  int gpu_obinsizex; // used for 3D spread block gather method
  int gpu_obinsizey;
  int gpu_obinsizez;

  int gpu_maxsubprobsize;
  int gpu_kerevalmeth;      // 0: direct exp(sqrt()), 1: Horner ppval

  int gpu_spreadinterponly; // 0: NUFFT, 1: spread or interpolation only

  int gpu_maxbatchsize;

  /* multi-gpu support */
  int gpu_device_id;

  void *gpu_stream;

  int modeord; // (type 1,2 only): 0 CMCL-style increasing mode order
               //                  1 FFT-style mode order

  int gpu_np;  // min batch_size for Output Driven

  int debug;   // 0: no debug, 1: debug
} cufinufft_opts;

#endif
