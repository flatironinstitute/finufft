.. _contft:

Efficient evaluation of continuous Fourier transforms of functions
======================================================================

Say you want to evaluate the *continuous Fourier transform* (FT) of a
given function, but you do not know the analytic formula for the FT.
You need a numerical evaluation method.
It is common to assume that the FFT is the right tool to do this,
but this rarely so ...unless you are
content with very poor accuracy!  The reason is that the FFT applies
only to equispaced data samples, which enforces the use of $N$ equispaced
nodes in any quadrature scheme for the Fourier integral.
Thus, unless you apply endpoint weight corrections (which are
available only in 1D, and stable only up to around 8th order),
you are generally stuck in 1D
with 1st or 2nd order (the standard trapezoid rule)
convergence with respect to $N$.
And there are many situations where a FFT-based scheme would be even worse:
this includes nonsmooth or singular functions (which demand custom
quadrature rules even in 1D), smooth functions with *varying length-scales*
(demanding *adaptive* quadrature for efficiency),
and possibly nonsmooth functions on complicated domains in higher dimensions.

Here we show that the NUFFT is often the right tool for
efficient and accurate Fourier
tranform evaluation, since it allows the user to apply
their favorite quadrature scheme
as appropriate for whatever nasty function they desire.
As long as $N$ is bigger than around 10, the NUFFT becomes more efficient
than direct evaluation of exponential sums; as we know, most quadrature
rules, especially in 2D or 3D, involve many more points than this.


1D Fourier transforms evaluated at arbitrary frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a function $f$, we'll need a single quadrature scheme with nodes
$x_j$ and weights $w_j$, $j=1,\dots, N$, that allows *accurate*
approximation of its Fourier integral

.. math:: \hat{f}(k) = \int f(x) e^{ikx} dx
   \approx \sum_{j=1}^N f(x_j) e^{ikx_j} w_j
   :label: fint

for all "target" frequencies $k$ in some domain of interest.
You can apply the below to any $f$ for which you have such a rule.

For simplicity let's take a smooth (somewhat oscillatory) function on $(a,b)$,
choose a million random frequency targets out to some ``kmax``,
and pick the Gauss-Legendre rule for $(a,b)$:

.. code-block:: matlab

  a=0; b=1;                     % interval
  f = @(x) cos(50*x.^2);        % our smooth function defined on (a,b), zero elsewhere
  M = 1e6;                      % # targets we want to compute the FT at
  kmax = 500;
  k = kmax * (2*rand(1,M)-1);   % desired target frequencies
  N = 200;                      % how many quadrature nodes
  [xj,wj] = lgwt(N,a,b);        % quadrature scheme for smooth funcs on (a,b)

Here is that function with the 200-node rule overlayed on it. You'll notice that the rule seems to be excessively fine (over-resolving $f(x)$), but that's because it actually needs to be able to resolve $f(x) e^{ikx}$ for all of our $k$ values:

.. image:: ../pics/contft1d.png
   :width: 100%

Notice :eq:`fint` is simply a type 3 NUFFT with strengths $c_j = f(x_j) w_j$,
so we evaluate it by calling FINUFFT (this takes 0.1 sec) then plot the resulting FT at its target $k$ points:

.. code-block:: matlab
           
  tol = 1e-10;
  fhat = finufft1d3(xj, f(xj).*wj, +1, tol, k);
  plot(k, [real(fhat),imag(fhat)], '.');

.. image:: ../pics/contft1dans.png
   :width: 100%

This looks like a continuous curve, but is actually (half a) million discrete points. How do we know to trust this answer? A convergence study in ``N`` shows that
200 nodes was indeed enough to reduce the quadrature error to below the
$10^{-10}$ NUFFT tolerance:

.. code-block:: matlab

  Ns = 100:10:220;             % N values to check convergence
  for i=1:numel(Ns), N=Ns(i);
    [xj,wj] = lgwt(N,a,b);     % N-node quadrature scheme for smooth funcs on (a,b)
    fhats{i} = finufft1d3(xj, f(xj).*wj, +1, tol, k);
  end
  f0 = norm(fhats{end},inf);   % compute rel sup norm of fhat vs highest-N case
  for i=1:numel(Ns)-1, errsup(i) = norm(fhats{i}-fhats{end},inf)/f0; end
  semilogy(Ns(1:end-1),errsup,'+-');

.. image:: ../pics/contft1dN.png
   :width: 60%

Remember: always do a convergence study!
We see rapid spectral convergence as the quadrature rule resolves the
oscillations in $e^{ikx}$ at $|k|=k_\text{max}$.
See `matlab/examples/contft1d.m <https://github.com/flatironinstitute/finufft/blob/master/matlab/examples/contft1d.m>`_ for the full code.

P.S. If you cared about only a few very high $k$ values,
`numerical steepest descent <https://users.flatironinstitute.org/~ahb/notes/numsteepdesc.html>`_ might eventually be best.


1D Fourier transforms evaluated on a uniform frequency grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the target frequencies lie on a uniform grid, the above type 3
NUFFT can be replaced by a faster type 1 NUFFT, by a simple rescaling.
Say that we replace the random targets in the above
example by the uniform grid

.. code-block:: matlab

   k = kmax * (-M:(M-1))/M;

***
   rescaled
   
Further reading:

* higher-order end corrections to the trapezoid rule:
  Kapur, S., Rokhlin, V. High-order corrected trapezoidal quadrature rules for singular functions. SIAM Journal on Numerical Analysis 34(4), 1331–1356 (1997)
