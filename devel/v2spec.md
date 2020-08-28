# FINUFFT v1.2 specifications -> now v2.0.

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

Here is a simple interface prototype, as of 6/29/20:

# simple/many-interface py calls... all of these would work:

f = finufft.nufft1d1(x,c,+1,1e-12,N)          # note that N must be given!
                                       # ntrans inferred from 2nd dim of c.
c = finufft.nufft1d2(x,f,+1,1e-12)          # type-2 size inferred from x,f.
                                       # ntrans inferred from 2nd dim of f.
f = finufft.nufft1d3(x,c,+1,1e-12,s)          # type-3 size inferred from x,s
                                       # ntrans inferred from 2nd dim of c.

f = finufft.nufft2d1(x,y,c,+1,1e-12,N1,N2)   # note that N1,N2 must be given!
finufft.nufft2d1(x,y,c,+1,1e-12,out=f)   # N1,N2 inferred from shape f
                                       # ntrans inferred from 2nd dim of c.

finufft.nufft1d1(x,c,+1,1e-12,out=f)        # now N can be inferred from shape f
                                    # error thrown if 2nd dim f != ntrans from c
finufft.nufft1d1(x,c,+1,1e-6,f)             # ditto
                                   # actually, how does this differ from out=..?
finufft.nufft1d1(x,c,+1,1e-6,f,modeord=0)    # trailing options
f = finufft.nufft1d1(x,c,+1,1e-6,modeord=0)    # trailing options (possible?)


Notes / discussion:

* I think it is crucial to keep tol as required argument.
This forces the user to understand that, unlike FFT and most math operations,
this is an approximate algorithm with user-chosen tolerance.
I have already seen FINUFFT usage in code saying things like
"Finufft has an accuracy of 1e-12", which is simply wrong. A default accuracy
would help propagate such myths in the numerically uneducated, so I am
against it.
Instead, forcing a required "set-accuracy" command is ok,
but clumsy (IMHO) since it's yet another custom interface routine.

* isign=+-1, defines the transform. This is related to having fft and
ifft commands (except not inverses). So I propose also forcing the
user to make this decision explicitly. It makes them read the formula.

* default mode-ordering - if we make python have FFT-ordering by default,
which is ok, will it be weird if the other languages don't?
The fortran users may still expect CMCL-compatibility. So we're stuck on a
fence. I'm willing to jump to FFT-default throughout all languages,
but have to update all the matlab docs now...


## Comparison against other NUFFT py interfaces:

https://github.com/pyNFFT/pyNFFT/blob/master/doc/source/tutorial.rst
is plan-type object-oriented interface.
Find rather clunky.
Forces the user to copy over their NU pts and data in a separate commands,
but why?

https://jyhmiinlin.github.io/pynufft/
is plan-type object-oriented interface.
has rather too many commands, but partly because iterative inverse xforms
are included. Also one controls upsampling and kernel width directly,
rather than tolerance (which forces the user to go too far into technical
details).



# implement simple interface via:
def finufft.nufft1d1(..., fout=None)
  throw error if fout wrong type/shape.
  If fout==None:
    create fout
  call guru
  return fout


Here's an example guru call, defining the guru interface (pre-6/29/20):
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



Use of out=None to write to returned array or to pre-alloc array in arg
list. See above.

pythonic error reporting

Joakim: if ordering is C not F, simply flip kx and ky pointers to NU locs.
(t1, t2 only, d=2 or d=3).


Old decisions:

* Decided not to detect whether "many" is called in guru (ie, n_transf>1)
via shape of input U array. This won't work for t3 many.
Instead force guru py user to give n_transf up front, in C++ guru.




Old decisions:

* Decided not to detect whether "many" is called in guru (ie, n_transf>1)
via shape of input U array. This won't work for t3 many.
Instead force guru py user to give n_transf up front, in C++ guru.





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
