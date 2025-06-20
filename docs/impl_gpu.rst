Implementation details
======================

This file contains detailed explanations of the algorithms and optimization strategies
used in the library.

The focus is on clarity and reproducibility of the core computational techniques,
including spreading/interpolation schemes, memory access patterns, and kernel launch
structures.

.. note::

   This is a living document. Implementation details are subject to change as
   performance and accuracy improvements are integrated.

Output Driven
-------------

The **output-driven spreading strategy** is designed to reduce global memory traffic and
exploit shared memory locality. A CUDA block corresponds to a spatial tile in the output
grid, and shared memory is used to accumulate updates from multiple nonuniform points.

The process follows three main stages:

1. **Per-thread kernel evaluation:**

   Each thread computes the spreading kernel at a single NUFFT point.
   Kernel values are stored into shared memory (`ker_evals`) in a batched layout,
   allowing reuse by all threads in the block.
   `ker_evals` is a 3D array with shape `(Np, dim, ns)`, where `Np` is the number of NUFFT points.
   Using CUDA parallelism it is possible to evaluate all the kernel values in parallel accessing
   `ker_evals(thread.id, dim, 0)`. The third parameter is always 0 because `eval_kernel_vec`
   takes a pointer and writes `ns` values in one go.
   This corresponds to:

   .. code-block:: cpp

      eval_kernel_vec<T, ns>(&ker_evals(i, 0, 0), x1, es_c, es_beta);

2. **Thread-cooperative accumulation in shared memory:**

   - Instead of assigning 1 thread per point (which would lead to shared memory collisions),
     all threads iterate over a small batch (`Np`) of NUFFT points.
     That is, the points are not processed in parallel, but the inner loop (tensor product) is.

     The **Shared Memory (SM) approach** does:

     .. code-block:: none

        parallel for point = 0 to NumPoints
          ...
          for x = 0 to ns
            for y = 0 to ns
              for z = 0 to ns
                ...

     The **Output-driven approach** does:

     For each point:

     - Loop over NUFFT points sequentially.
     - Parallelize over kernel grid entries using a flattened loop up to :math:`n_s^{\text{dim}}`.

     Example pseudocode:

     .. code-block:: none

        for point = 0 to NumPoints, point+=np
          ...
          parallel for i = 0 to pow(ns, dim)
            ...
          ...

     The parallelism is flipped: SM parallelizes the outer loop (over points), while
     Output-driven parallelizes the inner loop (over the kernel values).
     There is no collision because `local_subgrid` is accessed by `(ix, iy, iz)` — and these
     are unique per thread as determined by the thread ID.
     This removes the need for `AtomicAdd` on the local subgrid.

3. **Atomic addition to global memory:**

   Unchanged from SM: once all points have been processed and accumulated into `local_subgrid`,
   the block performs an atomic write to global memory (`fw`). Since this step is
   amortized over many points, its overhead is negligible.

Memory Organization
~~~~~~~~~~~~~~~~~~~

- `ker_evals`:
  Stores kernel weights in shape `(Np, dim, ns)`. Threads access only their assigned batch rows.

- `local_subgrid`:
  A padded shared-memory grid with shape :math:`(bin\_size + padding)^{dim}`.
  Where passing is :math:`padding = 2((ns+1)/2)`.
  Threads write to disjoint sections during accumulation to avoid races.

Design Insights
~~~~~~~~~~~~~~~

This hybrid parallelization combines **per-point parallelism** (step 1) and **spatial parallelism**
(step 2):

- Threads collaborate rather than compete on shared memory access.
- Batching (`Np`) controls memory footprint and allows tuning for hardware constraints.
- Synchronization barriers ensure correctness before accessing shared buffers.
