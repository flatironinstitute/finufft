# cufinufft

This is an implementation of nonuniform FFT on GPU.

### Code dependency
 - CUB library (https://github.com/NVlabs/cub)

### Usage
 - Get the CUB library - ```git clone https://github.com/NVlabs/cub.git```
 - Modify make.inc - set the ```INC``` with ```-I$(CUDA_DIR)/samples/common/inc/ -I$(CUDA_DIR)/include/ -I$(CUB_DIR)```
 - Compile - ```make all```
 - Run example - ``` ./cufinufft2d1_test 5 128 128 ```
 
