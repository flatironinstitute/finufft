Users and citations
===================

Here we list papers describing work which uses FINUFFT or its new spreading
kernel. Papers that merely cite our work are listed separately at the bottom. Please let me know if you are a user but not listed:

1. ASPIRE software for cryo-EM, based at Amit Singer's group at Princeton. https://github.com/PrincetonUniversity/ASPIRE-Python http://spr.math.princeton.edu/

#. "Cryo-EM reconstruction of continuous heterogeneity by Laplacian spectral volumes", Amit Moscovich, Amit Halevi, Joakim Andén, and Amit Singer. To appear, Inv. Prob. (2020), https://arxiv.org/abs/1907.01898

#. "A Fast Integral Equation Method for the Two-Dimensional Navier-Stokes Equations", Ludvig af Klinteberg, Travis Askham, and Mary Catherine Kropinski (2019), use FINUFFT 2D type 2. https://arxiv.org/abs/1908.07392

#. "MR-MOTUS: model-based non-rigid motion estimation for MR-guided radiotherapy using a reference image and minimal k-space data", Niek R F Huttinga, Cornelis A T van den Berg, Peter R Luijten and Alessandro Sbrizzi, Phys. Med. Biol. 65(1), 015004. https://arxiv.org/abs/1902.05776

#. Koga, K. "Signal processing approach to mesh refinement in simulations of axisymmetric droplet dynamics", https://arxiv.org/abs/1909.09553  Koga uses 1D FINUFFT to generate a "guideline function" for reparameterizing 1D curves.

#. L. Wang and Z. Zhao, "Two-dimensional tomography from noisy projection tilt
   series taken at unknown view angles with non-uniform distribution",
   International Conference on Image Processing (ICIP), (2019).

#. Aleks Donev's group at NYU.

Papers using our new window function but not the whole FINUFFT package:

1. Davood Shamshirgar and Anna-Karin Tornberg, "Fast Ewald summation for electrostatic potentials with arbitrary periodicity", exploit our "Barnett-Magland" (BM), aka exp-sqrt, window function. https://arxiv.org/abs/1712.04732

   #. Daniel Potts and Manfred Tasche, "Uniform error estimates for the NFFT", (2019) https://arxiv.org/abs/1912.09746 Claims a proof of tighter error constant for our exp-sqrt kernel, via Chebyshev expansions. See our remark on this in
      https://arxiv.org/abs/2001.09405
   
Citations to FINUFFT or its paper, that are not actual users:

1. https://arxiv.org/abs/1903.08365

#. https://arxiv.org/abs/1908.00041

#. https://arxiv.org/abs/1908.00574

