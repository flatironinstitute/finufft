.. finufft documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Flatiron Institute Nonuniform Fast Fourier Transform
====================================================

.. image:: logo.png
    :width: 45%
.. image:: spreadpic.png
    :width: 54%
	    
`FINUFFT <https://github.com/flatironinstitute/finufft>`_ is a multi-threaded library to compute efficiently the three most common types of nonuniform fast Fourier transform
(NUFFT) to a specified precision, in one, two, or three dimensions,
on a multi-core shared-memory machine.
It is extremely fast (typically achieving $10^6$ to $10^8$ points
per second),
has very simple interfaces to most major numerical languages
(C/C++, fortran, MATLAB, octave, python, and julia),
but also has more advanced (vectorized and "guru") interfaces that
allow multiple strength vectors and the reuse of FFT plans.
It is written in C++ (with limited use of ++ features), OpenMP, and calls
`FFTW <http://www.fftw.org>`_.
It has been developed at the `Center for Computational Mathematics
<https://www.simonsfoundation.org/flatiron/center-for-computational-mathematics/>`_ at the `Flatiron Institute <https://www.simonsfoundation.org/flatiron>`_,
by `Alex Barnett <https://users.flatironinstitute.org/~ahb>`_
and others, and is released under an
`Apache v2 license <https://github.com/flatironinstitute/finufft/blob/master/LICENSE>`_.

As an example, given $M$ arbitrary real numbers $x_j$ and complex
numbers $c_j$, with $j=1,\dots,M$, and a requested integer number of
modes $N$, FINUFFT can compute
the 1D type 1 (aka "adjoint") transform, which means it evaluates
the $N$ numbers

.. math:: f_k = \sum_{j=1}^M c_j e^{ik x_j}~, \qquad \mbox{ for } \; k\in\mathbb{Z}, \quad -N/2 \le k \le N/2-1 ~.
   :label: 1d1

As with other "fast" algorithms, FINUFFT does not evaluate this
sum directly (which would take $O(NM)$ effort),
but rather uses a sequence of steps (in this case, optimally chosen
spreading, FFT, and deconvolution stages)
to approximate the vector of answers :eq:`1d1` to within the user's
desired relative tolerance in (quasi-) *linear time*, ie, close to
$O(N+M)$ effort. Thus the speed-up is similar to that of the FFT.
For the two other transform types, and 2D and 3D cases, see
the :ref:`math <math>`_.

One interpretation of :eq:`1d1` is: the returned values $f_k$ are the
*Fourier series coefficients* of the $2\pi$-periodic
distribution $f(x) := \sum_{j=1}^M c_j \delta(x-x_j)$,
a sum of point-masses with arbitrary locations $x_j$ and strengths $c_j$.
Such exponential sums are needed in many applications in science and engineering, including signal processing (scattered data interpolation, applying convolutional transforms, fast summation), imaging (cryo-EM, CT, MRI gridding, coherent diffraction),
and numerical analysis
(computing Fourier *transforms* of functions,
moving between non-conforming quadrature grids,
solving partial differential equations).
See our :ref:`tutorials and demos<demos>` pages
and the :ref:`related works<related>`
for examples of how to use the NUFFT in applications.
In fact, there are many application areas where it has been overlooked
that the needed computation is simply a NUFFT
(eg, particle-mesh Ewald in molecular dynamics).

   
Why FINUFFT? Features and comparison against other NUFFT software
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic scheme used by FINUFFT is not new, but there are several
mathematical and implementation novelties that account for its high speed.
An upsampled (fine) grid underlies the calculation, that the user does
not need to have access to.
There is a tradeoff between the size of this grid, the size
of the spreading 
The upsampling



The FINUFFT library achieves its speed via several innovations including:

#. The use of a new spreading kernel that is provably close to optimal, yet faster to evaluate than the Kaiser-Bessel kernel
#. Quadrature approximation for the Fourier transform of the spreading kernel
#. Load-balanced multithreading of the type-1 spreading operation

   Rapid kernel evaluation via
piecewise polynomial approximation
that SIMD-vectorizes well.
  
point to spread pic above.


* vectorized 

* guru interface.
  
For the same accuracy in 3D, the
library is 3-50 times faster on a single core than the
single-threaded fast Gaussian gridding `CMCL libraries of Greengard-Lee <http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_, and in the multi-core setting
for spreading-dominated problems
is faster than the `Chemnitz NFFT3 library <https://www-user.tu-chemnitz.de/~potts/nfft/>`_ even when the latter is allowed a RAM-intensive full precomputation of the kernel. This is especially true for highly non-uniform point
distributions and/or high precision.
Our library does not require precomputation, and uses minimal RAM.

For the case of small problems where repeated NUFFTs are needed with a fixed set of nonuniform points, we have started to build advanced
interfaces for this case.
These are a factor of 2 or more faster than repeated calls to the plain
interface, since certain costs such as FFTW setup and sorting are performed
only once.

indebted to nfft, cmcl, for certain design aspects.


Do I need a NUFFT at all?
~~~~~~~~~~~~~~~~~~~~~~~~~

If you are new to this area, or even if not, it is important to
first



.. note::

   For very small repeated problems (less than 10000 input and output points),
   users should also consider a dense matrix-matrix multiplication against
   the NUDFT matrix using BLAS3 (eg ZGEMM). Since we did not want BLAS to
   be a dependency, we have not yet included this option.



   
   
.. toctree::
   :maxdepth: 2
	   
   install
   math
   dirs
   usage
   usage_adv
   error
   fortran          
   matlab
   pythoninterface
   juliainterface
   examples           
   related
   issues
   users
   ackn
   refs
   

   
