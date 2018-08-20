# cufinufft

This is an implementation of non-uniform FFT on GPU where the algorithms are based on FINUFFT (https://github.com/ahbarnett/finufft) a multi-threaded CPU code of non-uniform FFT developed by Alex H. Barnett and Jeremy F. Magland.

### Code dependency
 - CUB library (https://github.com/NVlabs/cub)

### Usage
 - Get the CUB library - ```git clone https://github.com/NVlabs/cub.git```
 - Modify make.inc - set the ```INC``` with ```-I$(CUDA_DIR)/samples/common/inc/ -I$(CUDA_DIR)/include/ -I$(CUB_DIR)```
 - Compile - ```make all```
 - Run example - ``` ./cufinufft2d1_test 5 128 128 ```

### Note
 - If you're running the code on GPU with Compute Capability less than 5.0 (ex. Kepler, Fermi), change the ```-arch=sm_50``` flag to lower number. (See http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) 
