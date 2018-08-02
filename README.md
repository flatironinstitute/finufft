# GPUnufftSpreader

This is an implementation of nufft spreader on GPU.

### Usage on ccblin019

```
  module load cuda
  make spread2d
  ./spread2d [nf1 nf2 [M [tol [Horner [use_thrust]]]]]
``` 

### Code dependency
 - sorting function in CUB library
 - scan in CUB library
 - sorting function in thurst

### To-do List
 - dir=2
 
#### 2018/06/29
 - Finish 1D dir=1
#### 2018/07/02 
 - Finish 2D dir=1
 - Add timing codes for comparison
#### 2018/07/05
 - Add input driven algorithm (this is also what've been done in cunfft)
#### 2018/07/17
 - Make both input/output algorithms works for large N1, N2, M
 - Add a hybrid method combining the idea of input/output driven algorithm
 - Sorted before doing an input driven spreading
 - Try replacing sort with thrust by cub library
#### 2018/07/18
 - Make hybrid method works for mod(nf1, bin_size_x)!=0 and mod(nf2, bin_size_y)!=0
 - put the function inside finufft (branch finufftwithgpuspreader)
#### 2018/07/19
 - Add accuracy check (compare the result with cnufftspread) 
#### 2018/07/20
 - Correct solution of finufft2d_test with gpu spreader
 - Add codes for speedup testing
#### 2018/08/02
 - Benchmark the performance for spreader of finufft, nfft, cufinufft, cunfft
 - Kernel time breakdown of all the methods
 - Add subproblem idea to deal with clustered nupts
