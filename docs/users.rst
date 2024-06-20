.. _users:

Dependent packages, wrappers, users, and citations
==================================================

Here we list packages that depend on or wrap FINUFFT, and papers or groups using it.
Papers that merely cite our work are listed separately at the bottom. Please let us know (and use github's dependent package link) if you are a user or package maintainer but not listed, and please "star" our GitHub repo.
It will help us to improve the library if you also
describe your use case parameters
`here <https://github.com/flatironinstitute/finufft/discussions/398>`_.


Packages relying on FINUFFT or cuFINUFFT
----------------------------------------

Here are some packages dependent on FINUFFT (please let us know of others,
and also add them to GitHub's Used By feature):

1. `SMILI <https://github.com/astrosmili/smili>`_, very long baseline interferometry reconstruction code by `Kazu Akiyama <http://kazuakiyama.github.io/>`_ and others, uses FINUFFT (2d1, 2d2, Fortran interfaces) as a `key library <https://smili.readthedocs.io/en/latest/install.html#external-libraries>`_. Akiyama used SMILI to reconstruct the `famous black hole image <https://physicstoday.scitation.org/do/10.1063/PT.6.1.20190411a/full/>`_ in 2019 from the Event Horizon Telescope.

#. `ASPIRE <http://spr.math.princeton.edu>`_: software for cryo-EM, based at Amit Singer's group at Princeton. `github <https://github.com/PrincetonUniversity/ASPIRE-Python>`_

#. `DISCUS <https://github.com/tproffen/DiffuseCode>`_: Fraunhofer diffraction from atomic clusters, nanomaterials and powders, by Reinhard Neder and others. Their manual (p.161) explains that FINUFFT enabled a two orders of magnitude speed-up.

#. `NWelch <https://github.com/sdrastro/NWelch>`_: Code for estimating power spectra and other properties of non-equispaced time-series, by astronomer Sarah Dodson-Robinson and others. Uses Python 1D type 3.

#. `Multitaper.jl <https://github.com/lootie/Multitaper.jl>`_: estimates power spectra from non-equispaced time series, improving upon Lomb-Scargle and NWelch, for exoplanet detection, by Sarah Dodson-Robinson and Charlotte Haley. Uses Julia 1D type 3.

#. `Pyxu <https://github.com/pyxu-org/pyxu>`_: Solves linear inverse problems with convex penalties using proximal optimization algorithms, in Python, by researchers at EPFL. This includes regularized imaging problems. (cu)FINUFFT is used for all `NUFFTs <https://pyxu-org.github.io/api/operator/linop.html#pyxu.operator.NUFFT>`_.

#. `MRI-NUFFT <https://mind-inria.github.io/mri-nufft/index.html>`_: unified Python interface to various NUFFT implementations for MRI reconstruction, with coil sensitivities, density compensation, and off-resonance corrections. From INRIA/CEA Paris Neurospin group.

#. `mri_distortion_toolkit <https://github.com/Image-X-Institute/mri_distortion_toolkit>`_: Characterisation and reporting of geometric distortion in MRI. Uses our PyPI pkg.

#. `EM-Align <https://github.com/ShkolniskyLab/emalign>`_: Aligning rotation, reflection, and translation between volumes (desntiy maps) in cryo-electron microscopy, from Shkolnisky Lab at Tel Aviv.

#. `spinifel <https://gitlab.osti.gov/mtip/spinifel>`_: Uses the multitiered iterative phasing (M-TIP) algorithm for single particle X-ray diffraction imaging, on CPU/GPU, from the ExaFEL project at LBNL/DOE.
   
#. `sinctransform <https://github.com/hannahlawrence/sinctransform>`_: C++ and MATLAB codes to evaluate sums of the sinc and sinc^2 kernels between arbitrary nonuniform points in 1,2, or 3 dimensions, by Hannah Lawrence (2017 summer intern at Flatiron).

#. `fsinc <https://github.com/gauteh/fsinc>`_:  Gaute Hope's fast sinc transform and interpolation Python package.

#. `FTK <https://github.com/flatironinstitute/ftk>`_: Factorization of the translation kernel for fast 2D rigid image alignment, by Rangan, Spivak, Andén, and Barnett.

#. `nifty-ls <https://github.com/flatironinstitute/nifty-ls>`_: Fast evaluation of the Lomb-Scargle periodogram for time series analysis, backed by finufft or cufinufft

#. `TRIQS CTINT <https://github.com/TRIQS/ctint>`_: continous time interaction-expansion solver, by N. Wentzell and O. Parcollet (Flatiron Institute, part of platform for interacting quantum systems).


Other wrappers to (cu)FINUFFT
------------------------------
   
#. `FINUFFT.jl <https://github.com/ludvigak/FINUFFT.jl>`_: a `julia <https://julialang.org/>`_ language wrapper by Ludvig af Klinteberg, Libin Lu, and others, now using pure Julia, and fully featured (rather than via Python). This is itself wrapped by `AbstractNFFTs.jl` in `NFFT.jl <https://juliamath.github.io/NFFT.jl/dev/performance/>`_.

#. `TensorFlow NUFFT <https://github.com/mrphys/tensorflow-nufft>`_: a wrapper to the differentiable machine learning Python tool TensorFlow, for the CPU (via FINUFFT) and GPU (via cuFINUFFT). By Javier Montalt Tordera (UCL).

#. `JAX bindings to (cu)FINUFFT <https://github.com/dfm/jax-finufft>`_: a wrapper to the differentiable machine learning Python tool JAX. Directly exposes the FINUFFT library to JAX's XLA backend, as well as implementing differentiation rules for the transforms. By Dan Foreman-Mackey (CCA).
   
#. `PyTorch wrapper to (cu)FINUFFT <https://flatironinstitute.github.io/pytorch-finufft>`_:  a wrapper to the differentiable machine learning Python tool PyTorch. By Michael Eickenberg and Brian Ward (CCM).
   

Research output using (cu)FINUFFT
---------------------------------

For the latest see: Google Scholar `FINUFFT citations <https://scholar.google.com/scholar?oi=bibs&hl=en&cites=14265215625340229167>`_, and `cuFINUFFT citations <https://scholar.google.com/scholar?oi=bibs&hl=en&cites=15739437776774999949>`_. Here are some highlights that we know about:

#. Marco Barbone and colleagues at Imperial have used FINUFFT and multi-GPU cuFINUFFT to accelerate 4D MRI reconstruction via the XD-GRASP algorithm by 10-20x. See their `2021 conference paper <https://ieeexplore.ieee.org/document/9651604>`_ and `2023 article <https://doi.org/10.1016/j.phro.2023.100484>`_.

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

#. The late Aleks Donev's group at NYU (Ondrej Maxian, et al) uses FINUFFT in Stokes viscous hydrodynamics solvers.

#. Efficient wide-field radio interferometry response. P. Arras, M. Reinecke, R. Westermann, T.A. Ensslin, Astron. Astrophys. (2020).   https://doi.org/10.1051/0004-6361/202039723

#. Johannes Blaschke, Jeff Donatelli, Jamie Sethian, and collaborators at the `ExaFEL <https://lcls.slac.stanford.edu/exafel>`_ coherent light source use FINUFFT and cuFINUFFT to accelerate single-particle X-ray imaging.  See preprint by Chang, Slaughter, Donatelli, et al: https://arxiv.org/abs/2109.05339

#. A. Harness, S. Shaklan, P. Willems, N. J. Kasdin, K. Balasubramanian, V. White, K. Yee, P. Dumont, R. Muller, S. Vuong, M. Galvin,
   "Optical experiments and model validation of perturbed starshade designs," Proc. SPIE 11823, Techniques and Instrumentation for Detection of Exoplanets X, 1182312 (1 September 2021); https://doi.org/10.1117/12.2595409

#. Chang, P., Pienaar, E., & Gebbie, T. (2020). "Malliavin--Mancino Estimators Implemented with Nonuniform Fast Fourier Transforms." SIAM J. Sci. Comput. 42(6), B1378–B1403. https://doi.org/10.1137/20m1325903 

#. Heisenberg voxelization (HVOX) for inteferometry of spherical sky maps in radio-astronomy, by Kashani, Simeoni, et al. (2023) https://arxiv.org/abs/2306.06007 https://github.com/matthieumeo/hvox

#. Sriramkrishnan Muralikrishnan at the Jülich Supercomputing Centre is running cufinufft on 6144 A100 GPUs (the NERSC-9 supercomputer), for a particle-in-Fourier method for plasma simulations. https://pasc23.pasc-conference.org/presentation/?id=msa167&sess=sess154

#. Related to that, FINUFFT is being used for a better-converging Fourier approach to the Immersed Boundary method of Peskin and his group at NYU. Zhe Chen and Charles Peskin, https://arxiv.org/abs/2302.08694
   
#. Pei R, Askham T, Greengard L, Jiang S (2023). "A fast method for imposing periodic boundary conditions on arbitrarily-shaped lattices in two dimensions." J. Comput. Phys. 474, 111792. https://doi.org/10.1016/j.jcp.2022.111792 Uses FINUFFT for plane wave sums.

#. Dylan Green, JR Jamora, and Anne Gelb (2023). "Leveraging joint sparsity in 3D synthetic aperture radar imaging," Appl. Math. Modern Chall. 1, 61-86. https://doi.org/10.3934/ammc.2023005 Uses 3D transforms between $N=201^3$ modes (voxels) and $M=313300$ data points. As they state, "...the computational cost of each method heavily depends on the NUFFT algorithm used."


Papers or codes using our new ES window (kernel spreading) function, but not the whole FINUFFT package:

1. Davood Shamshirgar and Anna-Karin Tornberg, "Fast Ewald summation for electrostatic potentials with arbitrary periodicity", exploit our "Barnett-Magland" (BM), aka exp-sqrt (ES) window function. https://arxiv.org/abs/1712.04732

#. Martin Reinecke: codes for radio astronomy reconstruction including https://gitlab.mpcdf.mpg.de/mtr/ducc

#. Aref Hashemi et al, "Computing hydrodynamic interactions in confined doubly-periodic geometries in linear time," J. Chem. Phys. 158(15): 154101 (2023).
DOI:10.1063/5.0141371.  https://arxiv.org/abs/2210.01837


Papers influenced by other aspects of FINUFFT:

1. NFFT.jl: Generic and Fast Julia Implementation of the Nonequidistant Fast Fourier Transform, by Tobias Knopp, Marija Boberg, Mirco Grosser (2022). https://arxiv.org/abs/2208.00049  They use our blocked spreading and piecewise polynomial ideas, and beat our type 1 and 2 performance by a factor of up to 1.7 in multithreaded cases. Code is dimension-independent but very abstract (two levels of meta-programming, I believe).

   
   
Some citations to FINUFFT that do not appear to be actual users
---------------------------------------------------------------

1. https://arxiv.org/abs/1903.08365

#. https://arxiv.org/abs/1908.00041

#. https://arxiv.org/abs/1908.00574

#. https://arxiv.org/abs/1912.09746

#. https://arxiv.org/abs/2010.05295
      
Now too many to track by hand... please see Google Scholar search linked above.
