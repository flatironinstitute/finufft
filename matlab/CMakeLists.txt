find_package(Matlab REQUIRED)
matlab_add_mex(NAME finufft_mex SRC finufft.cpp LINK_TO finufft OUTPUT_NAME finufft)
