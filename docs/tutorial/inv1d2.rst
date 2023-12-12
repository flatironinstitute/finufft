.. _inv1d2:

Inverse 1D type 2 NUFFT: fitting a Fourier series to scattered samples
======================================================================

This tutorial demonstrates inversion of the NUFFT using an iterative
solver and FINUFFT. For convenience it is in MATLAB/Octave.
The task is to solve for $N$ Fourier series coefficients $f_k$,
indexed by $-N/2 \le k < N/2$, given
samples $y_j$, $j=1,\dots,M$ of an unknown
$2\pi$-periodic function $f(x)$ at some given nodes $x_j$, $j=1,\dots,M$.
This has applications in signal analysis, as well
as being a 1D version of the MRI problem
(where the roles of real vs Fourier space are flipped).
Note that there are many other methods to fit smooth functions from
nonequispaced samples, eg high-order local Lagrange interpolation.
However, often your model for the function is a global Fourier series,
in which case, what we describe below is a good starting point.
We first assume that the samples are noise-free and no regularization is
needed, then proceed to the noisy regularized case.

Here's code to set up a random complex-valued
test problem of size 600000 by 300000 (way too large to solve directly):

.. code-block:: matlab

  N = 3e5;                             % how many unknown coeffs
  ks = -floor(N/2) + (0:N-1);          % row vec of the frequency k indices
  M = 2*N;                             % overdetermined by a factor 2
  x = 2*pi*rand(M,1);                  % scattered points on the periodic domain
  ftrue = randn(N,1) + 1i*randn(N,1);  % choose known Fourier coeffs at k inds
  ftrue = ftrue/sqrt(N);               % make signal f(x) variance=1, Re or Im part
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
We first evaluate the normal equations right-hand side via

.. code-block:: matlab

  rhs = finufft1d1(x,y,-1,tol,N);      % compute A^* y

We compare two ways to multiply $A^* A$ to a vector (perform the "matvec")
in the iterative solver.

**1) Matvec via a sequential pair of NUFFTs.** Here the matvec code is

.. code-block:: matlab

  function AHAf = applyAHA(f,x,tol)   % use pair of NUFFTs to apply A^*A to f
    Af = finufft1d2(x,+1,tol,f);                 % apply A
    AHAf = finufft1d1(x,Af,-1,tol,length(f));    % then apply A^*
  end

We target 6 digits from CG using this matvec function, then test the
residual and actual solution error:

.. code-block:: matlab

  [f,flag,relres,iter] = pcg(@(f) applyAHA(f,x,1e-6), rhs, 1e-6, N);
  norm(finufft1d2(x,+1,tol,f)-y)/norm(y)         % rel l2 resid for Af=y
  norm(f-ftrue)/norm(ftrue)                      % rel l2 coeff error
  
This reaches ``relres<1e-6`` in 1461 iterations
(a large count indicating poor conditioning),
taking about 100 seconds on an 8-core laptop.
The relative residual for the desired system $A{\bf f}={\bf y}$
is ``2.7e-05``, indicating that *the linear system was solved
reasonably accurately*,
but the relative coefficient error is a much larger
``2.4e-02``. Their ratio places a lower bound on the condition
number $\kappa(A)$ of about 900, explaining the large iteration count
for the normal equations.
Note that 0.0001% residual error in the normal equations resulted
in 2.4% coefficient error.

The error in the signal $f(x)$ is in fact very unequally distributed
for this problem: it is correct to 4-5 digits almost everywhere,
including at almost all the data points,
but errors are ${\cal O}(1)$ in the very largest gaps
between the (iid random) sample points:

.. image:: ../pics/inv1d2err.png
   :width: 90%

Notice the large error around 0.9212. However, the problem of
interpolating a band-limited function
is exponentially ill-conditioned with respect to the length of
any node-free gap measured in wavelengths. The gap near 0.9212 is
about 0.00009, ie, two wavelengths at the frequency $N/2$.
A sampling point distribution without large gaps would improve the conditioning
and make the reconstruction error in $f$ uniformly closer to the residual
error.
           
**2) Matvec exploiting Toeplitz structure via a pair of padded FFTs.**
A beautiful realization comes from examining the
usual matrix-matrix multiplication formula
for entries of the system matrix for the normal equations,

.. math:: (A^* A)_{k,k'} = \sum_{j=1}^M e^{i(k-k')x_j}
  \qquad \mbox{for } -N/2 \le k,k' < N/2.

We see the $k,k'$-entry only depends on $k-k'$, thus $A^*A$ is
Toeplitz (constant along diagonals). Its action on a vector is
thus a discrete convolution with a vector that we call $v$.



***

CG-Toep relres 9.97e-07 done in 1465 iters, 35 s


The solution and plot is essentially identical to that from the
NUFFT-pair method.
	rel l2 resid of Ax=y: 2.63e-05
	rel l2 coeff err: 0.0236




  
                

Further reading
---------------

For the 1D inversion with $M=N$ and no regularization
there are interpolation methods using
the fast multipole method for the cotangent kernel, eg:

*  A Dutt and V Rokhlin, Fast Fourier transforms for nonequispaced data, II. Appl. Comput. Harmonic Anal. 2, 85–100 (1995)

For the 2D iterative version using a Toeplitz matrix-vector multiply
for CG on the normal equations, in the MRI settings, see:

* J A Fessler et al,  Toeplitz-Based Iterative Image
  Reconstruction for MRI With Correction for Magnetic Field Inhomogeneity.
  IEEE Trans. Sig. Proc. 53(9) 3393 (2005).
