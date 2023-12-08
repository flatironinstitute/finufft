.. _inv1d2:

Fitting a Fourier series to scattered samples: inverse 1D type 2 NUFFT 
======================================================================

This tutorial demonstrates inversion of the NUFFT using an iterative
solver and FINUFFT. For convenience it is in MATLAB/Octave.
The task is to solve for $N$ Fourier series coefficients $f_k$,
indexed by $-N/2 \le k < N/2$, given
samples $y_j$, $j=1,\dots,M$ of an unknown
$2\pi$-periodic function at some given nodes $x_j$, $j=1,\dots,M$.
This has applications in signal analysis, as well
as being a 1D version of the MRI problem
(where the roles of real vs Fourier space are flipped).
Note that there are many other methods to fit smooth functions from
nonequispaced samples, eg high-order local Lagrange interpolation.
However, often your model for the function is a global Fourier series,
in which case, what we describe below is a good starting point.
We first assume that the samples are noise-free and no regularization is
needed, then proceed to the noisy regularized case.

Here's code to set up a small random test problem:

.. code-block:: matlab

  N = 1e5;                             % how many unknown coeffs
  ks = -floor(N/2) + (0:N-1);          % row vec of the frequency k indices
  M = 2*N;                             % overdetermined by a factor 2
  x = 2*pi*rand(M,1);                  % scattered points on the periodic domain
  ftrue = randn(N,1) + 1i*randn(N,1);  % choose known Fourier coeffs at k inds
  y = finufft1d2(x,+1,1e-12,ftrue);    % eval noiseless data at high accuracy


Case of no noise and no regularization
--------------------------------------

The linear system to solve is

.. math:: \sum_{-N/2\le k<N/2} e^{ik x_j} f_k = y_j
  \qquad \mbox{for } j=1,\dots,M.
  :label: linsys
          
This is usually formally overdetermined ($M>N$), although even in
that case it may be ill-conditioned
when the distribution of sample points $\{x_j\}$ has large gaps.
It is to be solved in the least-squares sense.
It is abbreviated by

.. math:: A{\bf f} = {\bf y}

where the $M\times N$ matrix has elements $A_{jk} = e^{ik x_j}$.
Left-multiplying by the conjugate $A^*$ gives the normal equations

.. math:: A^* A{\bf f} = A^* {\bf y}

where the system matrix $A^*A$ is symmetric positive definite,
so we use conjugate gradients (CG) to solve it iteratively.
The simplest way to apply $A^* A$ is by a pair of NUFFTs:

.. code-block:: matlab

  function AHAf = applyAHA(f,x,tol)   % use pair of NUFFTs to apply A^* A to vec
    Af = finufft1d2(x,+1,tol,f);                 % apply A
    AHAf = finufft1d1(x,Af,-1,tol,length(f));    % then apply A^*
  end

  *** bring in

  >> inv1d2
CG-NUFFT relres 9.94e-07 done in 1462 iters, 101 s
	rel l2 resid of Ax=y: 2.62e-05
	rel l2 coeff err: 0.0258
CG-Toep relres 9.97e-07 done in 1465 iters, 115 s
	rel l2 resid of Ax=y: 2.63e-05
	rel l2 coeff err: 0.0258
>> 

  
                

Further reading
~~~~~~~~~~~~~~~~

For the 1D inversion with $M=N$ and no regularization
there are interpolation methods using
the fast multipole method for the cotangent kernel, eg:

*  A Dutt and V Rokhlin, Fast Fourier transforms for nonequispaced data, II. Appl. Comput. Harmonic Anal. 2, 85–100 (1995)

For the 2D iterative version using a Toeplitz matrix-vector multiply
for CG on the normal equations, in the MRI settings, see:

* J A Fessler et al,  Toeplitz-Based Iterative Image
  Reconstruction for MRI With Correction for Magnetic Field Inhomogeneity.
  IEEE Trans. Sig. Proc. 53(9) 3393 (2005).
