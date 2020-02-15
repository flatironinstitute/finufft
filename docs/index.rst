.. finufft documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Flatiron Institute Nonuniform Fast Fourier Transform
====================================================

.. image:: logo.png
    :width: 45%
.. image:: spreadpic.png
    :width: 54%
	    
`FINUFFT <https://github.com/flatironinstitute/finufft>`_ is a set of libraries to compute efficiently three types of nonuniform fast Fourier transform
(NUFFT) to a specified precision, in one, two, or three dimensions,
on a multi-core shared-memory machine.
The library has a very simple interface, does not need any precomputation step,
is written in C++ (using OpenMP and FFTW),
and has wrappers to C, fortran, MATLAB, octave, and python.
As an example, given $M$ arbitrary real numbers $x_j$ and complex
numbers $c_j$, with $j=1,\dots,M$, and a requested integer number of
modes $N$, the 1D type-1 (aka "adjoint") transform evaluates the $N$ numbers

.. math:: f_k = \sum_{j=1}^M c_j e^{ik x_j}~, \qquad \mbox{ for } \; k\in\mathbb{Z}, \quad -N/2 \le k \le N/2-1 ~.
   :label: 1d1

The $x_j$ can be interpreted as nonuniform source locations, $c_j$
as source strengths, and $f_k$ then as the $k$th Fourier series coefficient
of the distribution $f(x) = \sum_{j=1}^M c_j \delta(x-x_j)$.
Such exponential sums are needed in many applications in science and engineering, including signal processing, imaging, diffraction, and numerical 
partial differential equations.
The naive CPU effort to evaluate :eq:`1d1` is $O(NM)$.
The library approximates :eq:`1d1` to a requested relative precision
$\epsilon$ with nearly linear effort $O(M \log (1/\epsilon) + N \log N)$.
Thus the speedup over the naive cost is similar to that achieved by the FFT.
This is achieved by spreading onto a regular grid using a carefully chosen kernel,
followed by an upsampled FFT, then a division (deconvolution) step.
For the 2D and 3D definitions, and other types of transform, see below.

The FINUFFT library achieves its speed via several innovations including:

#. The use of a new spreading kernel that is provably close to optimal, yet faster to evaluate than the Kaiser-Bessel kernel
#. Quadrature approximation for the Fourier transform of the spreading kernel
#. Load-balanced multithreading of the type-1 spreading operation

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
   matlab
   pythoninterface
   juliainterface
   examples           
   related
   issues
   users
   ackn
   refs
   

   
