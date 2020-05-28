# cuFINUFFT
A GPU implementation of 2, 3 dimension type 1, 2 non-uniform FFT based on [FINUFFT][1].

This is a work from Melody Shih's internship at Flatiron Institute, advised by CCM project leader Alex Barnett.


### Installation
 - Get this code and dependency -
 ```git clone https://github.com/flatironinstitute/cufinufft.git```
 - Review the `makefile`. - If you need to customize build settings, create and edit a `make.inc`.  Example:
   - To override the standard CUDA `/usr/local/cuda` location your `make.inc` should contain: ```CUDA_ROOT=/your/path/to/cuda```.
   - Two examples are provided, one for IBM machines (make.inc.power9), and another for the Courant Institute cluster (make.inc.CIMS).
 - Compile - ```make all -j```
 - Run a test code - ``` ./cufinufft2d1_test 2 128 128 10 1e-6```
 
### Interface
cuFINUFFT API contains 5 stages:
 - Set cufinufft default options - ```int ier=cufinufft_default_opts(type1, dim, dplan.opts);```
 - Make cufinufft plan - ``` ier=cufinufft_makeplan(type1, dim, nmodes, iflag, ntransf, tol, maxbatchsize, &dplan); ```
 - Set the locations of non-uniform points x,y,z - ```ier=cufinufft_setNUpts(M, x, y, z, 0, NULL, NULL, NULL, &dplan);```
 - Apply the transformation with data c,fk - ```ier=cufinufft_exec(c, fk, &dplan); ```
 - Destroy cufinufft plan - ```ier=cufinufft_destroy(&dplan);```
 
### Preprocessors
 - SINGLE - single-precision.  Enable by calling `PREC=SINGLE make ...` or adding to your make.inc.
 - TIME - timing for each stage.  Enable by adding "-DTIME" to `NVCCFLAGS`.
 - SPREADTIME - more detailed timing from spreading and interpolation
 - DEBUG - debug mode outputs all the middle stages' result
 
### Other
 - If you are interested in optimizing for GPU Compute Capability,
 you may want to specicfy ```NVARCH=-arch=sm_XX``` in your make.inc to reduce compile times,
 or for other performance reasons. See [Matching SM Architectures][2].

[1]: https://github.com/flatironinstitute/finufft
[2]: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
