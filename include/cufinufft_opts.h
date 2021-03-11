#ifndef __CUFINUFFT_OPTS_H__
#define __CUFINUFFT_OPTS_H__

typedef struct cufinufft_opts {   // see cufinufft_default_opts() for defaults
	double upsampfac;   // upsampling ratio sigma, only 2.0 (standard) is implemented
	/* following options are for gpu */
        int gpu_method;  // 1: nonuniform-pts driven, 2: shared mem (SM)
	int gpu_sort;    // when NU-pts driven: 0: no sort (GM), 1: sort (GM-sort)

	int gpu_binsizex; // used for 2D, 3D subproblem method
	int gpu_binsizey;
	int gpu_binsizez;

	int gpu_obinsizex; // used for 3D spread block gather method
	int gpu_obinsizey;
	int gpu_obinsizez;

	int gpu_maxsubprobsize;
	int gpu_nstreams;
	int gpu_kerevalmeth; // 0: direct exp(sqrt()), 1: Horner ppval

	int gpu_spreadinterponly; // 0: NUFFT, 1: spread or interpolation only

	/* multi-gpu support */
	int gpu_device_id;
} cufinufft_opts;

#endif
