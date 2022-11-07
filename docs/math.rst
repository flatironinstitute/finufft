.. _math:

Mathematical definitions of transforms
======================================

We use notation with a general space dimensionality $d$, which will
be 1, 2, or 3, in our library.
The arbitrary (ie nonuniform) points in space are denoted
$\mathbf{x}_j \in \mathbb{R}^d$, $j=1,\dots,M$.
We will see that for type 1 and type 2, without loss of generality
one could restrict to the periodic box $[-\pi,\pi)^d$.
For type 1 and type 3, each such NU point carries a given associated strength
$c_j\in\mathbb{C}$.
Type 1 and type 2 involve the Fourier "modes" (Fourier series coefficients)
with integer frequency indices lying in the Cartesian product set

.. math::
   
   K = K_{N_1,\dots,N_d} := K_{N_1} \times K_{N_2} \times \dots \times K_{N_d}~,

where

.. math::

  K_{N_i} := \left\{\begin{array}{ll} \{-N_i/2,\ldots,N_i/2-1\}, & N_i \mbox{ even},\\
  \{-(N_i-1)/2,\ldots,(N_i-1)/2\}, & N_i \mbox{ odd}.
  \end{array}\right.

For instance, $K_{10} = \{-5,-4,\dots,4\}$,
whereas $K_{11} = \{-5,-4,\dots,5\}$,
and so $K_{10,11} = \{(-5,-5),(-4,-5),\dots,(4,-5),(-5,-4),(-4,-4),\dots,(3,5),(4,5)\}$.
Note that the ordering in the last case is with the first index "fast", second
"slow"; this matches the storage ordering in the library interface.
Thus, in the 1D case $K$ is an interval containing $N_1$ integer indices,
in 2D it is
a list of $N_1N_2$ index pairs (which may be thought of as a rectangle of frequencies), and in 3D it is a list of $N_1N_2N_3$
index triplets (which may be thought of as a cuboid).

Then the **type 1** (nonuniform to uniform, aka "adjoint") NUFFT evaluates

.. math::
  :label: 1
   
  f_\mathbf{k} := \sum_{j=1}^M c_j e^{\pm i \mathbf{k}\cdot \mathbf{x}_j}
  \qquad \mbox{for } \mathbf{k} \in K

  
This can be viewed as evaluating a set of
Fourier series coefficients due to sources
with strengths $c_j$ at the arbitrary locations $\mathbf{x}_j$.	  
Either sign of the imaginary unit in the exponential can be chosen in the interface. Note that our normalization differs from that of references [DR,GL].

The **type 2** (U to NU, aka "forward") NUFFT evaluates

.. math::
   :label: 2
	   
   c_j := \sum_{\mathbf{k}\in K} f_\mathbf{k} e^{\pm i \mathbf{k}\cdot \mathbf{x}_j}
   \qquad \mbox{for } j=1,\ldots,M


This is the adjoint of the type 1, ie the evaluation of a given Fourier
series at a set of arbitrary points.
Both type 1 and type 2 transforms are invariant under
translations of the NU points by multiples of $2\pi$,
thus one could require that all NU points live in the
origin-centered box $[-\pi,\pi)^d$.
In fact, as a compromise between library speed, and flexibility for the user
(for instance, to avoid boundary points being flagged as outside of
this box due to round-off error), our library only
requires that the NU points lie in the three-times-bigger box
$\mathbf{x}_j \in [-3\pi,3\pi]^d$.
This allows the user to choose a convenient periodic domain that does not
touch this three-times-bigger box.
However, there may be a slight speed increase if most points fall in
$[-\pi,\pi)^d$.

Finally, the **type 3** (NU to NU) transform does not have restrictions on
the NU points, and there is no periodicity.
Let $\mathbf{x}_j\in\mathbb{R}^d$, $j=1,\ldots,M$, be NU locations, with strengths $c_j \in \mathbb{C}$,
and let $\mathbf{s}_k$, $k=1,\ldots,N$ be NU frequencies.
Then the type 3 transform evaluates:

.. math::
  :label: 3
   
  f_k := \sum_{j=1}^M c_j e^{\pm i \mathbf{s}_k\cdot \mathbf{x}_j}
   \qquad \mbox{for } k=1,\ldots,N

For all three transforms, the computational effort scales like the
product of the space-bandwidth products (real-space width times frequency-space width) in each dimension. For type 1 and type 2 this means near-linear
scaling in the total number of modes $N := N_1\dots N_d$.
However, be warned that for type 3 this means that, even if $N$ and $M$ are
small, if the product of the tightest intervals enclosing the coordinates of
$\mathbf{x}_j$ and $\mathbf{s}_k$ is large, the algorithm will be
inefficient. For such NU points, a direct sum should be used instead.


We emphasise that the NUFFT tasks that this library performs
should not be confused with either the discrete Fourier transform (DFT),
the (continuous) Fourier transform (although it may be used to approximate
this via a quadrature rule), or the inverse NUFFT (the iterative solution of
the linear system arising from nonuniform Fourier sampling, as in, eg, MRI).
It is also important to know that, for NU points, *the type 1 is not
the inverse of the type 2*.
See the references for clarification.
