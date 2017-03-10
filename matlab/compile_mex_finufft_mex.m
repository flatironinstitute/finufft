mex ./_mcwrap/mcwrap_finufft1d1_mex.cpp ./finufft_mex.cpp -output ./finufft1d1_mex ../lib/libfinufft.a -lm -lgomp -lfftw3 -lfftw3_threads -lrt
mex ./_mcwrap/mcwrap_finufft2d1_mex.cpp ./finufft_mex.cpp -output ./finufft2d1_mex ../lib/libfinufft.a -lm -lgomp -lfftw3 -lfftw3_threads -lrt
mex ./_mcwrap/mcwrap_finufft3d1_mex.cpp ./finufft_mex.cpp -output ./finufft3d1_mex ../lib/libfinufft.a -lm -lgomp -lfftw3 -lfftw3_threads -lrt
