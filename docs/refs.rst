.. _refs:

References
==========

Please cite the following two papers if you use this software:

[FIN]
A parallel non-uniform fast Fourier transform library based on an ``exponential of semicircle'' kernel.
A. H. Barnett, J. F. Magland, and L. af Klinteberg.
SIAM J. Sci. Comput. 41(5), C479-C504 (2019). `arxiv version <https://arxiv.org/abs/1808.06736v2>`_

[B20]
Aliasing error of the exp$(\beta \sqrt{1-z^2})$ kernel in the nonuniform fast Fourier transform.
A. H. Barnett. submitted, Appl. Comput. Harmon. Anal. (2020).
`arxiv version <https://arxiv.org/abs/2001.09405>`_

Background references
~~~~~~~~~~~~~~~~~~~~~

For the Kaiser--Bessel kernel and the related PSWF, see:

[KK] Chapter 7. System Analysis By Digital Computer. F. Kuo and J. F. Kaiser. Wiley (1967).

[FT]
K. Fourmont. Schnelle Fourier-Transformation bei nichtäquidistanten Gittern und tomographische Anwendungen. PhD thesis, Univ. Münster, 1999.

[F] Non-equispaced fast Fourier transforms with applications to tomography.
K. Fourmont.
J. Fourier Anal. Appl.
9(5) 431-450 (2003).

[FS] Nonuniform fast Fourier transforms using min-max interpolation.
J. A. Fessler and B. P. Sutton. IEEE Trans. Sig. Proc., 51(2):560-74, (Feb. 2003)

[ORZ] Prolate Spheroidal Wave Functions of Order Zero: Mathematical Tools for Bandlimited Approximation.  A. Osipov, V. Rokhlin, and H. Xiao. Springer (2013).

[KKP] Using NFFT3---a software library for various nonequispaced fast Fourier transforms. J. Keiner, S. Kunis and D. Potts. Trans. Math. Software 36(4) (2009).

[DFT] How exponentially ill-conditioned are contiguous submatrices of the Fourier matrix? A. H. Barnett, submitted, SIAM Rev. (2020).
`arxiv version <https://arxiv.org/abs/2004.09643>`_

The appendix of the last of these contains the first known published proof
of the Kaiser--Bessel Fourier transform pair.

FINUFFT builds upon the CMCL NUFFT, and the Fortran wrappers are very similar to its interfaces. For that, the following are references:

[GL] Accelerating the Nonuniform Fast Fourier Transform. L. Greengard and J.-Y. Lee. SIAM Review 46, 443 (2004).

[LG] The type 3 nonuniform FFT and its applications. J.-Y. Lee and L. Greengard. J. Comput. Phys. 206, 1 (2005).

The original NUFFT analysis using truncated Gaussians is (the second
improving upon the first):

[DR] Fast Fourier Transforms for Nonequispaced data. A. Dutt and V. Rokhlin. SIAM J. Sci. Comput. 14, 1368 (1993).

[S] A note on fast Fourier transforms for nonequispaced grids.
G. Steidl, Adv. Comput. Math. 9, 337-352 (1998).
