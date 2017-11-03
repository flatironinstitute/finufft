Contents of the package
=======================

 `src` : main library source and headers. Compiled .o will be built here  

`lib` : compiled library will be built here  

`makefile` : GNU makefile (user may need to edit)  

`test` : validation and performance tests. `test/check_finufft.sh` is the main validation script. `test/nuffttestnd.sh` is the main performance test script  

`test/results` : validation comparison outputs (\*.refout; do not remove these), and local test and performance outputs (\*.out; one may remove these)  

`examples` : simple example codes for calling the library from C++ and from C  

`fortran` : wrappers and drivers for Fortran  

`matlab` : wrappers and examples for MATLAB/octave  

`finufftpy` : python wrappers  

`python_tests` : accuracy and speed tests and examples using the python wrappers  

`contrib` : 3rd-party code  

`doc` : contains the manual  

`setup.py` : enables building and installing of the python wrappers using pip or pip3  

`README.md` : this file  

`INSTALL.md` : installation instructions for various operating systems  

`LICENSE` : how you may use this software  

`CHANGELOG` : list of changes, release notes  

`TODO` : list of things needed to do, or wishlist  
