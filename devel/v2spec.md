# FINUFFT v1.2 specifications

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


finufft_makeplan is passed ptr to opts object, or NULL which uses defaults.

[should finufft_makeplan return a plan object, or a pointer to plan? no.
Instead it needs an error code, so return that. A ptr to plan is an arg.]



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



## Python interface

Here's an example guru call, defining the guru interface:
This is close to Joakim's GPU interface at
https://github.com/janden/cufinufft/blob/python/python/cufinufft.py

# type 1 or 2 ------------
import finufft
N = (50,100)     # shape implicitly sets dimension, here 2D
type = 1
isign = +1
tol = 1e-6
plan = finufft.plan(type, N, ntransf, isign, tol)     # there's a last optional arg opts=None
  # if opts=None then use default opts  (opts lives inside plan)
# (set x,y)
finufft.set_nu_pts(plan, x,y)     # here y=None, z=None in the def
finufft.execute(plan, c, f)    # either reads c and writes to existing f, or opposite.
f = finufft.execute(plan, c)    # another way to call (eg for type 1), creates f
finufft.destroy(plan)

# type 3 ----------------
N=2    # now N is interpreted as the dim ? or have separate dim argument??
plan = finufft.plan(3, N, ntransf, isign, tol)
# (set x,y, sx, sy)
finufft.set_nu_pts(plan, x,y,None, sx,sy,None)  # here sx,sy,sz are output NU  
finufft.execute(plan, c, f)    # reads c and writes to f.
# note c, f size must match ntransf
finufft.destroy(plan)

# this needs to be fleshed out.

# simple py calls:
outf = finufft.nufft1d1(...)
# implement via:
def finufft.nufft1d1(..., fout=None)
  throw error if fout wrong.
  If fout==None:
    create fout
  return fout




Notes:

There will be pure-python wrappers implementing
the simple finufftpy.nufft() calls.
INTERFACE TO DO.

proceed with pybind11. Allow python to see and edit all fields in opts struct.

Notes on finufftpy.cpp:
// * pyfinufft -> finufftpy everywhere,  right?
// * do we need this cpp module at all - can we interface directly to guru
//   cmds in the C++ lib?

[Pass ptr to plan, but py user cannot see inside it. ?  ie, "blind pointer"
No: copy Joakim's GPU interface plan?
]

Detect whether "many" is called in guru (ie, n_transf>1) via shape
of input U array. This won't work for t3 many. See above.

Use of out=None to write to returned array or to pre-alloc array in arg
list. See above.

pythonic error reporting

Joakim: if ordering is C not F, simply flip kx and ky pointers to NU locs.
(t1, t2 only, d=2 or d=3).





## Directory structure

src
include
contrib
lib
lib-static
test   - directft
       - results
                *.refout
examples (C++/C examples)
python - finufftpy (the py module)
       - examples
       - test
       setup.py    (this is for pybind11)
       requirements.txt
fortran - test
                dir*.f
        - examples
                *demo*.f
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
