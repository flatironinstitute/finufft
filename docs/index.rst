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
by :ref:`Alex Barnett and others<ackn>`,
and is released under an
`Apache v2 license <https://github.com/flatironinstitute/finufft/blob/master/LICENSE>`_.

What does FINUFFT do?
~~~~~~~~~~~~~~~~~~~~~

As an example, given $M$ arbitrary real numbers $x_j$ and complex
numbers $c_j$, with $j=1,\dots,M$, and a requested integer number of
modes $N$, FINUFFT computes
the 1D type 1 (aka "adjoint") transform, which means it evaluates
the $N$ numbers

.. math:: f_k = \sum_{j=1}^M c_j e^{ik x_j}~, \qquad \mbox{ for } \; k\in\mathbb{Z}, \quad -N/2 \le k \le N/2-1 ~.
   :label: 1d1

As with other "fast" algorithms, FINUFFT does not evaluate this
sum directly---which would take $O(NM)$ effort---but
rather uses a sequence of steps (in this case, optimally chosen
spreading, FFT, and deconvolution)
to approximate the vector of answers :eq:`1d1` to within the user's
desired relative tolerance with only $O(N \log N +M)$ effort,
ie, quasi-linear. Thus the speed-up is similar to that of the FFT.
You may want to jump to :ref:`quickstart <quick>`, or see
the :ref:`definitions <math>` of the type 2 and 3 transforms,
and 2D and 3D cases.

One interpretation of :eq:`1d1` is: the returned values $f_k$ are the
*Fourier series coefficients* of the $2\pi$-periodic
distribution $f(x) := \sum_{j=1}^M c_j \delta(x-x_j)$,
a sum of point-masses with arbitrary locations $x_j$ and strengths $c_j$.
Such exponential sums are needed in many applications in science and engineering, including signal processing (scattered data interpolation, applying convolutional transforms, fast summation), imaging (cryo-EM, CT, MRI gridding, coherent diffraction),
and numerical analysis
(computing Fourier *transforms* of functions,
moving between non-conforming quadrature grids,
solving partial differential equations).
See our :ref:`tutorials and demos<tut>` pages
and the :ref:`related works<related>`
for examples of how to use the NUFFT in applications.
In fact, there are many application areas where it has been overlooked
that the needed computation is simply a NUFFT
(eg, particle-mesh Ewald in molecular dynamics).

Why FINUFFT? Features and comparison against other NUFFT libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic scheme used by FINUFFT is not new, but there are many
mathematical and software engineering improvements over other libraries.
As is common in NUFFT algorithms, under the hood is an FFT on a
regular "fine" (upsampled) grid---the user
has no need to access this directly. Nonuniform points are either spread to,
or interpolated from, this fine grid, using a specially designed kernel
(see right figure above).
Our main features are:

* **High speed**.  For instance, at similar accuracy, FINUFFT is up to 10x faster than the multi-threaded `Chemnitz NFFT3 library <https://www-user.tu-chemnitz.de/~potts/nfft/>`_, and (in single-thread mode) up to 50x faster than the `CMCL NUFFT library <http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_. This is achieved via:

  1. a simple new `"exponential of semicircle" kernel <https://arxiv.org/abs/2001.09405>`_ that is provably close to optimal
  #. quadrature approximation for this kernel's Fourier transform
  #. load-balanced multithreaded spreading/interpolation (see left figure above)
  #. bin-sorting of points to improve cache reuse
  #. a low upsampling option for smaller FFTs, especially in type 3 transforms
  #. piecewise polynomial kernel evaluation (additions and multiplications only) that SIMD-vectorizes reliably on open-source compilers

* **Less RAM**. Our kernel is so fast that there is no point in precomputation; it is always evaluated on the fly. Thus our memory footprint is often an order of magnitude less than the fastest (precomputed) modes of competitors such as NFFT3 and MIRT, especially at high accuracy.

* **Automated kernel parameters**. Unlike many competitors, we do not force the user to worry about kernel choice or parameters. The user simply requests a desired relative accuracy, then FINUFFT chooses parameters that achieve this accuracy as fast as possible.

* **Simplicity**. We have simple interfaces that perform a NUFFT with a single command---just like an FFT---from seven common languages/environments. For advanced users we also have "many vector" interfaces that can be much faster than repeated calls to the simple interface with the same points. Finally (like NFFT3) we have a "guru" interface for maximum flexibility, in all of these languages.

For technical details on much of the above see our :ref:`papers <refs>`.
Note that there are other tasks (eg, transforms on spheres, inverse NUFFTs)
provided by other libraries, such as NFFT3, that FINUFFT does not provide.


.. _need:

Do I even need a NUFFT?
~~~~~~~~~~~~~~~~~~~~~~~

Maybe you already know that your application needs a NUFFT.
For instance, if you need Fourier transforms or power spectra
but have data on non-equispaced grids, you may be able to
rewrite your task as one of the :ref:`three transform types<math>`.
To help decide, see the :ref:`tutorials and demos<tut>`.
If so, and both $M$ and $N$ are larger than of order $10^2$, FINUFFT may
be the ticket.
However, if $M$ and/or $N$ is small (of order $10$ or less)
you should simply evaluate the sums directly.
Another scenario is that you wish to evaluate, eg, :eq:`1d1` repeatedly with
the same set of nonuniform points $x_j$ but *fresh* strength vectors
$\{c_j\}_{j=1}^M$, as in the "many vectors" interface mentioned above.
In that case it may be better to fill the $N$-by-$M$ matrix $A$ with entries
$a_{kj} = e^{ik x_j}$, then use BLAS3 (eg ``ZGEMM``) to compute $F = AC$,
where each column of $F$ and $C$ is a new instance of :eq:`1d1`.
If you have very many columns this can be competitive with a NUFFT
even for $M$ and $N$ up to $10^4$, because BLAS3 is so fast.

Contents of remainder of documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
.. toctree::
   :maxdepth: 2
	   
   install
   dirs
   math
   c
   opts
   error
   trouble
   tut
   fortran          
   matlab
   pythoninterface
   juliainterface
   related
   users
   ackn
   refs
   

   
