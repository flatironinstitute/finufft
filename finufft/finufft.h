#ifndef FINUFFT_H
#define FINUFFT_H

struct nufft_opts {   // see common/finufft_default_opts() for defaults
	FLT upsampfac;      // upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)
	/* following options are for gpu */
	int gpu_method;
	int gpu_sort; // used for 3D nupts driven method

	int gpu_binsizex; // used for 2D, 3D subproblem method
	int gpu_binsizey;
	int gpu_binsizez;

	int gpu_obinsizex; // used for 3D spread block gather method
	int gpu_obinsizey;
	int gpu_obinsizez;

	int gpu_maxsubprobsize;
	int gpu_nstreams; 
	int gpu_kerevalmeth;	// 0: direct exp(sqrt()), 1: Horner ppval
};
#endif
