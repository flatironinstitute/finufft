.. _users:

Dependent packages, wrappers, users, and citations
==================================================

Here we list packages that depend on or wrap FINUFFT, and papers or groups using it.
Papers that merely cite our work are listed separately at the bottom. Please let us know (and use github's dependent package link) if you are a user or package maintainer but not listed, and please "star" our GitHub repo.


Packages relying on FINUFFT or cuFINUFFT
----------------------------------------

Here are some packages dependent on FINUFFT (please let us know of others,
and also add them to github's Used By feature):

1. `SMILI <https://github.com/astrosmili/smili>`_, very long baseline interferometry reconstruction code by `Kazu Akiyama <http://kazuakiyama.github.io/>`_ and others, uses FINUFFT (2d1, 2d2, Fortran interfaces) as a `key library <https://smili.readthedocs.io/en/latest/install.html#external-libraries>`_. Akiyama used SMILI to reconstruct the `famous black hole image <https://physicstoday.scitation.org/do/10.1063/PT.6.1.20190411a/full/>`_ in 2019 from the Event Horizon Telescope.

#. `ASPIRE <http://spr.math.princeton.edu>`_: software for cryo-EM, based at Amit Singer's group at Princeton. `github <https://github.com/PrincetonUniversity/ASPIRE-Python>`_

#. `sinctransform <https://github.com/hannahlawrence/sinctransform>`_: C++ and MATLAB codes to evaluate sums of the sinc and sinc^2 kernels between arbitrary nonuniform points in 1,2, or 3 dimensions, by Hannah Lawrence (2017 summer intern at Flatiron).

#. `fsinc <https://github.com/gauteh/fsinc>`_:  Gaute Hope's fast sinc transform and interpolation Python package.

#. `FTK <https://github.com/flatironinstitute/ftk>`_: Factorization of the translation kernel for fast 2D rigid image alignment, by Rangan, Spivak, Andén, and Barnett.

#. `DISCUS <https://github.com/tproffen/DiffuseCode>`_: Fraunhofer diffraction from atomic clusters, nanomaterials and powders, by Reinhard Neder and others. Their manual explains that FINUFFT enabled a two orders of magnitude speed-up.

#. `NWelch <https://github.com/sdrastro/NWelch>`_: Code for estimating power spectra and other properties of non-equispaced time-series, by astronomer Sarah Dodson-Robinson and others.

#. `Picsou <https://github.com/matthieumeo/pycsou>`_: package for solving linear inverse problems with convex penalties using proximal optimization algorithms, in Python, by researchers at EPFL. This includes regularized imaging problems.
   
   
Other wrappers to (cu)FINUFFT
------------------------------
   
#. `FINUFFT.jl <https://github.com/ludvigak/FINUFFT.jl>`_: a `julia <https://julialang.org/>`_ language wrapper by Ludvig af Klinteberg, Libin Lu, and others, now using pure Julia, and fully featured (rather than via Python). This is itself wrapped by `AbstractNFFTs.jl` in `NFFT.jl <https://juliamath.github.io/NFFT.jl/dev/performance/>`_.

#. `TensorFlow NUFFT <https://github.com/mrphys/tensorflow-nufft>`_: a wrapper to the differentiable machine learning Python tool TensorFlow, for the CPU (via FINUFFT) and GPU (via cuFINUFFT). By Javier Montalt Tordera (UCL).

#. `JAX bindings to FINUFFT <https://github.com/dfm/jax-finufft>`_: a wrapper to the differentiable machine learning Python tool JAX. Directly exposes the FINUFFT library to JAX's XLA backend, as well as implementing differentiation rules for the transforms. By Dan Foreman-Mackey (CCA).
   
   

Research output using (cu)FINUFFT
---------------------------------

For the latest see: Google Scholar `FINUFFT citations <https://scholar.google.com/scholar?oi=bibs&hl=en&cites=14265215625340229167>`_, and `cuFINUFFT citations <https://scholar.google.com/scholar?oi=bibs&hl=en&cites=15739437776774999949>`_. Here are some early highlights we know about:

#. Marco Barbone has used FINUFFT and cuFINUFFT to accelerate 4D MRI reconstruction via the XD-GRASP algorithm by 11x. See their `2021 conference paper <https://ieeexplore.ieee.org/document/9651604>`_.

#. "Cryo-EM reconstruction of continuous heterogeneity by Laplacian spectral volumes", Amit Moscovich, Amit Halevi, Joakim Andén, and Amit Singer. Appeared in Inv. Prob. (2020), https://arxiv.org/abs/1907.01898

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

#. Aleks Donev's group at NYU; ongoing.

#. Efficient wide-field radio interferometry response. P. Arras, M. Reinecke, R. Westermann, T.A. Ensslin, Astron. Astrophys. (2020).   https://doi.org/10.1051/0004-6361/202039723

#. Johannes Blaschke, Jeff Donatelli, and collaborators at NSERC/LBNL use FINUFFT and cuFINUFFT for single-particle X-ray imaging.

#. A. Harness, S. Shaklan, P. Willems, N. J. Kasdin, K. Balasubramanian, V. White, K. Yee, P. Dumont, R. Muller, S. Vuong, M. Galvin,
   "Optical experiments and model validation of perturbed starshade designs," Proc. SPIE 11823, Techniques and Instrumentation for Detection of Exoplanets X, 1182312 (1 September 2021); https://doi.org/10.1117/12.2595409

#. Chang, P., Pienaar, E., & Gebbie, T. (2020). "Malliavin--Mancino Estimators Implemented with Nonuniform Fast Fourier Transforms." SIAM J. Sci. Comput. 42(6), B1378–B1403. https://doi.org/10.1137/20m1325903 

#. Heisenberg voxelization (HVOX) for inteferometry of spherical sky maps in radio-astronomy, by Kashani, Simeoni, et al. (2023) https://arxiv.org/abs/2306.06007 https://github.com/matthieumeo/hvox



Papers or codes using our new ES window (spreading) function but not the whole FINUFFT package:

1. Davood Shamshirgar and Anna-Karin Tornberg, "Fast Ewald summation for electrostatic potentials with arbitrary periodicity", exploit our "Barnett-Magland" (BM), aka exp-sqrt (ES) window function. https://arxiv.org/abs/1712.04732

#. Martin Reinecke: codes for radio astronomy reconstruction including https://gitlab.mpcdf.mpg.de/mtr/ducc

#. S. Jiang and L. Greengard,
   new multilevel kernel-split faster 3D FMM.


Papers influenced by other aspects of FINUFFT:

1. NFFT.jl: Generic and Fast Julia Implementation of the Nonequidistant Fast Fourier Transform, by Tobias Knopp, Marija Boberg, Mirco Grosser (2022). https://arxiv.org/abs/2208.00049  They use our blocked spreading and piecewise polynomial ideas, and beat our type 1 and 2 performance by a factor of 1-2 in some cases.

   
   
Some citations to FINUFFT that do not appear to be actual users
---------------------------------------------------------------

1. https://arxiv.org/abs/1903.08365

#. https://arxiv.org/abs/1908.00041

#. https://arxiv.org/abs/1908.00574

#. https://arxiv.org/abs/1912.09746

#. https://arxiv.org/abs/2010.05295
      
Now too many to track by hand... see Google Scholar search above.
