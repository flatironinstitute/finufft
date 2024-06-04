.. _nfft_migr:

Migration guide from NFFT3
==========================

Here we outline how the C/C++ user can replace NUFFT calls to the popular library
`Chemnitz NFFT3 library <https://www-user.tu-chemnitz.de/~potts/nfft/>`_ with
FINUFFT CPU calls to achieve the same effect, possibly with more performance
or less RAM usage.
See [KKP] in the :ref:`references<refs>` for more about NFFT3, and [FIN] for
some performance and RAM comparisons performed using the codes available in 2018.
We use the `NFFT source on GitHub <https://github.com/NFFT/nfft>`_, version 3.5.4alpha.
So far we only discuss:

 * the adjoint NFFT (a.k.a. type 1)

Also of interest (but not yet demonstrated below) is:

 * the forward NFFT transform (a.k.a. type 2)
 * the nonuniform to nonuniform NNFFT (a.k.a. type 3)
   
 .. note:: The NFFT3 library can do more things---real-valued data, sphere, rotation group, hyperbolic cross, inverse transforms---none of which FINUFFT can yet do directly (although our three transforms can be used as components in such tasks). We do not address those here.

Migrating a 2D adjoint transform (type 1) in C from NFFT3 to FINUFFT
--------------------------------------------------------------------

We need to start with the simplest example of using NFFT3 on "user data" generated
using plain, transparent, C commands (rather than relying on NFFT3-supplied
data-generation, direct transform, and printing utilities as in the
NFFT example :file:`examples/nfft/simple_test.c`, or even its simplest
version
at https://www-user.tu-chemnitz.de/~potts/nfft/download/nfft_simple_test.tar.gz ).
We choose 2D since it is the simplest rectangular
case that illustrates how to get the transposed
coordinates (or mode array) ordering correct.
After installing NFFT3 one should be able to compile and run the following:

.. literalinclude:: ../tutorial/nfft2d1_test.c
   :language: c

This is a basic example, running single-threaded, at the highest precision
(using ``nfft_init_guru`` would allow more control.)
It demonstrates: i) the NFFT3 user must write their data into arrays allocated
by ``nfft_plan``,
ii) the single nonuniform point coordinate array
is interleaved ($x_1, y_1, x_2, y_2, \dots, x_M, y_M$),
iii) the output mode ordering is C (row-major) rather than Fortran (column-major;
this affects how to convert frequency indices into the output array index),
and iv) there is an extra factor of $2\pi$ in the exponent relative
to the FINUFFT definition, because NFFT3 assumes a 1-periodic input domain.
The code is found in our :file:`tutorial/nfft2d1_test.c`. Running the executable gives:

::

 2D type 1 (NFFT3) done in 0.589 s: f_hat[-17,33]=86.0632804289+-350.023846367i, rel err 9.93e-14

To show how to migrate this, we write a self-contained code that generates exactly
the same "user data" (same random seed), then uses FINUFFT to do the transform
to achieve exactly the same ``f_hat`` output array (in row-major C ordering).
This entails scaling and swapping the nonequispaced coordinates just before sending
to FINUFFT. Here is the corresponding C code (compare to the above):

.. literalinclude:: ../tutorial/migrate2d1_test.c
   :language: c

The fact that NFFT3 uses row-major mode arrays whereas FINUFFT uses column-major has
been handled here by swapping the input $x$ and $y$ coordinates and array sizes in the
FINUFFT call. (Equivalently, this could have been achieved by transposing the ``f_hat``
output array. We recommend the former route since it saves memory.) Running the
executable gives:

::

 2D type 1 (FINUFFT) in 0.0787 s: f_hat[-17,33]=86.0632804289+-350.023846367i, rel err 9.58e-14

Comparing to the above, we see the same answer to all shown digits, a similar error for this tested output entry, plus a 7.5$\times$ speed-up. (Both use a single thread, tested on the same AMD 5700U laptop.) The user may of course now set a coarser (larger) value for ``tol`` and see a further speed-up.

We believe that the above gives the essentials of how to convert your code from using NFFT3 to FINUFFT. Please read our documentation, especially the guru interface if multiple related transforms are required, then post a GitHub Issue if you are still stuck.
