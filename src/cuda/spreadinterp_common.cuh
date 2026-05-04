// Shared helpers used by multiple per-method spread/interp .cuh bodies.
//
// Extracted to deduplicate the bin/subproblem metadata setup that is
// identical across spread_subprob, interp_subprob, and spread_output_driven
// (kernel side) and across the prep_* drivers (host side). The functions
// are __forceinline__/inline so codegen is unchanged from the inlined
// originals.

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

// Per-block metadata read by every subprob-style kernel: which bin this
// block serves, where its NU points start in the bin-sorted array, how
// many of them there are, and the local-grid offset for the bin.
template<typename T, int ndim> struct subprob_block_info {
  cuda::std::array<int, 3> binsizes;
  cuda::std::array<int, 3> nbins;
  cuda::std::array<int, ndim> offset;
  int bidx;
  int ptstart;
  int nupts;
};

template<typename T, int ndim>
__device__ __forceinline__ subprob_block_info<T, ndim> compute_subprob_block_info(
    const cufinufft_gpu_data<T> &p, int subpidx) {
  cuda::std::array<int, 3> binsizes{p.opts.gpu_binsizex, p.opts.gpu_binsizey,
                                    p.opts.gpu_binsizez};
  auto nbins            = get_nbins<ndim>(p.nf123, binsizes);
  const int bidx        = loadReadOnly(p.subprob_to_bin + subpidx);
  const int binsubp_idx = subpidx - loadReadOnly(p.subprobstartpts + bidx);
  const int ptstart =
      loadReadOnly(p.binstartpts + bidx) + binsubp_idx * p.opts.gpu_maxsubprobsize;
  const int nupts =
      min(p.opts.gpu_maxsubprobsize,
          loadReadOnly(p.binsize + bidx) - binsubp_idx * p.opts.gpu_maxsubprobsize);
  auto offset = compute_offset<ndim>(bidx, nbins, binsizes);
  return {binsizes, nbins, offset, bidx, ptstart, nupts};
}

// Host-side bin-layout bundle used by prep_nupts_driven and
// prep_subprob_and_OD. Same four values are derived from p.opts and
// p.nf123 in both places.
template<typename T, int Ndim> struct bin_layout {
  cuda::std::array<int, 3> binsizes;
  cuda::std::array<int, 3> nbins;
  int nbins_tot;
  cuda::std::array<T, 3> inv_binsizes;
};

template<typename T, int Ndim>
inline bin_layout<T, Ndim> compute_bin_layout(
    const cufinufft_opts &opts, const cuda::std::array<CUFINUFFT_BIGINT, 3> &nf123) {
  cuda::std::array<int, 3> binsizes{opts.gpu_binsizex, opts.gpu_binsizey,
                                    opts.gpu_binsizez};
  auto nbins      = get_nbins<Ndim>(nf123, binsizes);
  const int total = nbins_total(nbins);
  cuda::std::array<T, 3> inv{T(1) / binsizes[0], T(1) / binsizes[1], T(1) / binsizes[2]};
  return {binsizes, nbins, total, inv};
}

} // namespace spreadinterp
} // namespace cufinufft
