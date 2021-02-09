.. _users:

Dependent packages, users, and citations
========================================

Here we list packages that depend on FINUFFT, and papers or groups using it.
Papers that merely cite our work are listed separately at the bottom. Please let us know (and use github's dependent package link) if you are a user or package maintainer but not listed.

Packages relying on FINUFFT
---------------------------

Here are some packages dependent on FINUFFT (please let us know of others,
and also add them to github's Used By feature):

1. `SMILI <https://github.com/astrosmili/smili>`_, very long baseline interferometry reconstruction code by `Kazu Akiyama <http://kazuakiyama.github.io/>`_ and others, uses FINUFFT (2d1, 2d2, Fortran interfaces) as a `key library <https://smili.readthedocs.io/en/latest/install.html#external-libraries>`_. Akiyama used SMILI to reconstruct the `famous black hole image <https://physicstoday.scitation.org/do/10.1063/PT.6.1.20190411a/full/>`_ in 2019 from the Event Horizon Telescope.

#. `ASPIRE <http://spr.math.princeton.edu>`_: software for cryo-EM, based at Amit Singer's group at Princeton. `github <https://github.com/PrincetonUniversity/ASPIRE-Python>`_

#. `sinctransform <https://github.com/hannahlawrence/sinctransform>`_: C++ and MATLAB codes to evaluate sums of the sinc and sinc^2 kernels between arbitrary nonuniform points in 1,2, or 3 dimensions, by Hannah Lawrence (2017 summer intern at Flatiron).

#. `fsinc <https://github.com/gauteh/fsinc>`_:  Gaute Hope's fast sinc transform and interpolation python package.

#. `FTK <https://github.com/flatironinstitute/ftk>`_: Factorization of the translation kernel for fast rigid image alignment, by Rangan, Spivak, Andén, and Barnett.
      
#. `FINUFFT.jl <https://github.com/ludvigak/FINUFFT.jl>`_: a `julia <https://julialang.org/>`_ language wrapper by Ludvig af Klinteberg (SFU), now using pure julia rather than python.



Research output using FINUFFT
-----------------------------

#. "Cryo-EM reconstruction of continuous heterogeneity by Laplacian spectral volumes", Amit Moscovich, Amit Halevi, Joakim Andén, and Amit Singer. To appear, Inv. Prob. (2020), https://arxiv.org/abs/1907.01898

#. "A Fast Integral Equation Method for the Two-Dimensional Navier-Stokes Equations", Ludvig af Klinteberg, Travis Askham, and Mary Catherine Kropinski, J. Comput. Phys., 409 (2020) 109353; uses FINUFFT 2D type 2. https://arxiv.org/abs/1908.07392

#. "MR-MOTUS: model-based non-rigid motion estimation for MR-guided radiotherapy using a reference image and minimal k-space data", Niek R F Huttinga, Cornelis A T van den Berg, Peter R Luijten and Alessandro Sbrizzi, Phys. Med. Biol. 65(1), 015004. https://arxiv.org/abs/1902.05776

#. Koga, K. "Signal processing approach to mesh refinement in simulations of axisymmetric droplet dynamics", https://arxiv.org/abs/1909.09553  Koga uses 1D FINUFFT to generate a "guideline function" for reparameterizing 1D curves.

#. L. Wang and Z. Zhao, "Two-dimensional tomography from noisy projection tilt
   series taken at unknown view angles with non-uniform distribution",
   International Conference on Image Processing (ICIP), (2019).

#. "Factorization of the translation kernel for fast rigid image alignment,"
   Aaditya Rangan, Marina Spivak, Joakim Andén, and Alex Barnett.
   Inverse Problems 36 (2), 024001 (2020).
   https://arxiv.org/abs/1905.12317

#. Aleks Donev's group at NYU; ongoing

#. Efficient wide-field radio interferometry response. P. Arras, M. Reinecke, R. Westermann, T.A. Ensslin, Astron. Astrophys. (2020).   https://doi.org/10.1051/0004-6361/202039723

#. Johannes Blaschke, Jeff Donatelli, and collaborators at NSERC/LBNL use FINUFFT and cuFINUFFT for single-particle X-ray imaging.

Papers or codes using our new ES window (spreading) function but not the whole FINUFFT package:

1. Davood Shamshirgar and Anna-Karin Tornberg, "Fast Ewald summation for electrostatic potentials with arbitrary periodicity", exploit our "Barnett-Magland" (BM), aka exp-sqrt (ES) window function. https://arxiv.org/abs/1712.04732

#. Martin Reinecke: codes for radio astronomy reconstruction including https://gitlab.mpcdf.mpg.de/mtr/ducc


   
Citations to FINUFFT that do not appear to be actual users
----------------------------------------------------------

1. https://arxiv.org/abs/1903.08365

#. https://arxiv.org/abs/1908.00041

#. https://arxiv.org/abs/1908.00574

#. https://arxiv.org/abs/1912.09746

#. https://arxiv.org/abs/2010.05295
      
