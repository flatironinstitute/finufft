# FINUFFT v1.2 specifications

2/6/20

We will write the docs for v1.2 first, then fill out implementation in a
fresh branch.
Developing in guruInterface_v2 is fine for now.

When stuff is moved over to fresh branch,
don't forget to commit --author="AKMalleo3" etc to preserve the author.



## Usage from C++ or C

main change is to pass *pointer* to opts always; this allows NULL as default

int ier = finufft1d1(M,x,c,+1,tol,N,F,NULL);       // takes default opts

nufft_opts opts;
finufft_default_opts(&opts);
opts.debug = 2;
int ier = finufft1d1(M,x,c,+1,tol,N,F,&opts);       // general opts

Here nufft_opts is a simple struct, not an object.


### guru

finufft_makeplan is passed ptr to opts object, or NONE:
It starts with

if (opts_ptr==NONE) {
  nufft_opts *opts_ptr;
  finufft_default_opts(opts_ptr);
}





### implementation of finufft1d (as in "legacy" but should be in src)

```
int finufft1d1(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts_ptr)
{
  ... (note no finufft_type)
  ...
  int ier = invokeGuruInterface(n_dims, 1, n_transf, nj, xj, NULL, NULL, cj,
				iflag, eps, n_modes, 0, NULL, NULL, NULL, fk,
                                opts_ptr);
  return ier;
}

int invokeGuruInterface(int n_dims, int type, int n_transf, BIGINT nj, 
FLT* xj,FLT *yj, FLT *zj, CPX* cj,int iflag, FLT eps, BIGINT *n_modes, BIGINT nk, FLT *s, FLT *t,  FLT *u,  CPX* fk, nufft_opts *opts_ptr)
{
  finufft_plan plan;
  ...
  int ier = finufft_makeplan(type, n_dims, n_modes, iflag, n_transf, eps, blksize, &plan, opts_ptr);
  (rest as in guruInterface branch...)
}


```

## Usage from python

proceed with pybind11. Allow python to see and edit all fields in opts struct.

Pass ptr to plan, but py user cannot see inside it.

Detect whether "many" is called in guru (ie, n_transf>1) via shape
of input U array.

Use of out=None to write to returned array or to pre-alloc array in arg
list.

pythonic error reporting

Joakim: if ordering is C not F, simply flip kx and ky pointers to NU locs.
(t1, t2 only, d=2 or d=3).





## Directory structure

src
include
contrib
lib
lib-static
test   - direct
examples
python - finufftpy
       - examples
       - test
       setup.py
       requirements.txt
fortran
matlab - examples
       - test
       *.m (interfaces)
docs
devel
LICENSE
CHANGELOG
TODO
README.md
finufft-manual.pdf
make.inc.*
