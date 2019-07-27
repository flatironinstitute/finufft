# cuFINUFFT
A GPU implementation of 2,3 dimension type 1,2 non-uniform FFT based on FINUFFT (https://github.com/flatironinstitute/finufft). This is a work as a summer intern at Flatiron Institute advised by CCM project leader Alex Banett.


### Code dependency
 - CUB library (https://github.com/NVlabs/cub)

### Installation
 - Get the CUB library - ```git clone https://github.com/NVlabs/cub.git```
 - Modify make.inc - set the ```INC``` with ```-I$(CUDA_DIR)/samples/common/inc/ -I$(CUDA_DIR)/include/ -I$(CUB_DIR)```
 - Compile - ```make all```
 - Run a test code - ``` ./cufinufft2d1_test 2 128 128 10 1e-6```
 
### Interface
cuFINUFFT API contains 5 stages:
 - Set cufinufft default options - ```int ier=cufinufft_default_opts(type1, dim, dplan.opts);```
 - Make cufinufft plan - ``` ier=cufinufft_makeplan(type1, dim, nmodes, iflag, ntransf, tol, ntransfcufftplan, &dplan); ```
 - Set the locations of non-uniform points x,y,z - ```ier=cufinufft_setNUpts(M, x, y, z, 0, NULL, NULL, NULL, &dplan);```
 - Apply the transformation with data c,fk - ```ier=cufinufft_exec(c, fk, &dplan); ```
 - Destroy cufinufft plan - ```ier=cufinufft_destroy(&dplan);```
 
### Preprocessors
 - SINGLE - single-precision
 - TIME - timing for each stage
 - SPREADTIME - more detailed timing from spreading and interpolation
 - DEBUG - debug mode outputs all the middle stages' result
 
### Other
 - If you're running the code on GPU with Compute Capability less than 5.0 (ex. Kepler, Fermi), change the ```-arch=sm_50``` flag to lower number. (See http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) 
