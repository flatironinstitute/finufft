.. _refs:

References
==========

Please cite the first two papers if you use the CPU library FINUFFT, and the third if you use the GPU library cuFINUFFT:

[FIN]
A parallel non-uniform fast Fourier transform library based on an "exponential of semicircle" kernel.
A. H. Barnett, J. F. Magland, and L. af Klinteberg.
SIAM J. Sci. Comput. 41(5), C479-C504 (2019). `arxiv version <https://arxiv.org/abs/1808.06736>`_

[B20]
Aliasing error of the exp$(\beta \sqrt{1-z^2})$ kernel in the nonuniform fast Fourier transform.
A. H. Barnett. Appl. Comput. Harmon. Anal. 51, 1-16 (2021).
`arxiv version <https://arxiv.org/abs/2001.09405>`_

[S21]
cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs.
Y.-H. Shih, G. Wright, J. Andén, J. Blaschke, and A. H. Barnett.
PDSEC2021 workshop of the IPDPS2021 conference (*best paper prize* at workshop). `arxiv version <https://arxiv.org/abs/2102.08463>`_

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

The appendix of the last of the above contains the first known published proof
of the Kaiser--Bessel Fourier transform pair.
This next two papers prove error estimates for sinh-type and other kernels closely related (and possibly slightly more optimal) than ours:

[PT] Uniform error estimates for the NFFT. D. Potts and M. Tasche. (2020). `arxiv <https://arxiv.org/abs/1912.09746v2>`_

[PT2] Continuous window functions for NFFT.  D. Potts and M. Tasche. (2020). `arxiv <https://arxiv.org/abs/2010.06894>`_. In revision, Adv. Comput. Math.

In late 2020 it was pointed out to us by Piero Angeletti that the exponential of semicircle kernel developed for FINUFFT had in fact been independently proposed:

[AN] A new window based on exponential function. K. Avci and A. Nacaroğlu. 2008 Ph.D. Research in Microelectronics and Electronics, Istanbul. 69-72 (2008). doi:10.1109/RME.2008.4595727.

FINUFFT builds upon the CMCL NUFFT, and the Fortran wrappers are very similar to its interfaces. For that, the following are references:

[GL] Accelerating the Nonuniform Fast Fourier Transform. L. Greengard and J.-Y. Lee. SIAM Review 46, 443 (2004).

[LG] The type 3 nonuniform FFT and its applications. J.-Y. Lee and L. Greengard. J. Comput. Phys. 206, 1 (2005).

Inversion of the NUFFT is covered in [KKP] above, and in:

[GLI] The fast sinc transform and image reconstruction from nonuniform samples in $\mathbf{k}$-space. L. Greengard, J.-Y. Lee and S. Inati, Commun. Appl. Math. Comput. Sci (CAMCOS) 1(1) 121-131 (2006).

The original NUFFT analysis using truncated Gaussians is (the second
improving upon the convergence rate of the first):

[DR] Fast Fourier Transforms for Nonequispaced data. A. Dutt and V. Rokhlin. SIAM J. Sci. Comput. 14, 1368 (1993).

[S] A note on fast Fourier transforms for nonequispaced grids.
G. Steidl, Adv. Comput. Math. 9, 337-352 (1998).

Talk slides
~~~~~~~~~~~

These
`PDF slides <http://users.flatironinstitute.org/~ahb/talks/pacm20.pdf>`_
may be a useful introduction to FINUFFT and applications.

Yu-Hsuan (Melody) Shih's PDSEC2021 20-minute presentation video on cuFINUFFT is here: https://www.youtube.com/watch?v=PnW6ehMyHxM

