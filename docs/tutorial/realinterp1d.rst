.. _realinterp1d:

Fast Fourier interpolation of 1D real-valued function at arbitrary points
=========================================================================

This is a Python variant of the previous 1D MATLAB demo, but illustrating
a trick for **real-valued** functions.
For real-valued functions the Fourier series coefficients
``-N//2`` to ``(N-1)//2`` arising from the FFT of ``N`` regular samples
have Hermitian symmetry,
so that it could be considered wasteful to send all ``N`` coefficients
into a type-2 NUFFT to evaluate the Fourier series at new targets.
Here we show that it is possible to half the length of the
NUFFT input array (and thus internal FFT size), increasing
efficiency if the FFTs are dominant. A similar trick (again saving a factor of two
in FFT cost, not two per dimension) could be done in 2D or 3D.

Let our periodic domain be $[0,2\pi)$.
We sample a bandlimited function that is exactly captured by $N$
point samples, use the real-valued FFT to get its coefficients
with nonnegative indices, then show how to interpolate this Fourier
series to a set of random target points. The problem sizes are kept
deliberately small (see other demos which scale things up):

.. code-block:: python

  import numpy as np
  import finufft as fi

  N = 5                                                # num regular samples
  # a generic real test fun  with bandlimit <= (N-1)/2 so interp exact...
  fun = lambda t: 1.0 + np.sin(t+1) + np.sin(2*t-2)    # bandlimit is 2

  Nt = 100                                             # test targs
  targs = np.random.rand(Nt)*2*np.pi

  Nf = N//2 + 1                                        # num freq outputs for rfft
  g = np.linspace(0,2*np.pi,N,endpoint=False)          # sample grid
  f = fun(g)
  c = (1/N) * np.fft.rfft(f)   # gets coeffs 0,1,..Nf-1  (don't forget prefac)
  assert c.size==Nf

  # Do the naive (double-length c array) NUFFT version:
  cref = np.concatenate([np.conj(np.flip(c[1:])), c])  # reflect to 1-Nf...Nf-1 coeffs
  ft = np.real(fi.nufft1d2(targs,cref,eps=1e-12,isign=1))       # f at targs (isign!)
  # (taking Re here was just a formality; it is already real to eps_mach)
  print("naive (reflected) 1d2 max err:", np.linalg.norm(fun(targs) - ft, np.inf))

  # now demo avoid doubling the NUFFT length via freq shift and mult by phase:
  c[1:] *= 2.0     # since each nonzero coeff appears twice in reflected array
  N0 = Nf//2       # starting freq index shift that FINUFFT interprets for c array
  ftp = fi.nufft1d2(targs,c,eps=1e-12,isign=1)         # f at targs but with phase
  # the key step: rephase (to account for shift), only then take Re (needed!)...
  ft = np.real( ftp * (np.cos(N0*targs) + 1j*np.sin(N0*targs)))   # guess 1j sign
  print("unpadded 1d2 max err:", np.linalg.norm(fun(targs) - ft, np.inf))

When run this gives:

.. code-block::

  naive (reflected) 1d2 max err: 9.898748487557896e-13
  unpadded 1d2 max err: 6.673550601021816e-13

which shows that both schemes work.
See the full code `tutorial/realinterp1d.py <https://github.com/flatironinstitute/finufft/blob/master/tutorial/realinterp1d.py>`_.
This arose from Discussion https://github.com/flatironinstitute/finufft/discussions/720


.. note::

    Complex-valued spreading/interpolation is still used under the hood in FINUFFT, so that there is no
    efficiency gain on the nonuniform point side,
    possibly motivating real-valued NUFFT variants. Since the decisions about
    real-valued interfaces become elaborate, we leave this for future work.

Notes about the 2D case:
``numpy.rfft2`` seems to have a different output
size than ``rfft``.
One can still only save a factor of two in the ``nufft2d2`` input array size (hence FFT size), just as in 1D. It would need a rectangular (``N/2`` by ``N``) array, where ``N ``is the number of regular samples per dimension.
Special handling the origin and the zero-index cases will be needed
to recreate the effect of the full reflected Hermitian-symmetric coefficient array. Please contribute a demo.
