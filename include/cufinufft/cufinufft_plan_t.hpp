#ifndef CUFINUFFT_PLAN_T_H
#define CUFINUFFT_PLAN_T_H

#include <cuda/std/array>
#include <cufft.h>
#include <cufinufft/defs.hpp>
#include <cufinufft/types.hpp>
#include <cufinufft_opts.h>
#include <finufft_common/spread_opts.h>
#include <finufft_errors.h>
#include <optional>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <type_traits>

#include <cstddef>
#include <cuComplex.h>
#include <memory>
#include <tuple>

/* This header file contains the internal plan class of cufinufft,
   as well as other types and functions which are exclusively
   needed in combination with that class. */

// RAII wrapper for cufftHandle (which is a plain int, not a pointer, so
// std::unique_ptr does not adapt to it cleanly). The destructor body is
// defined in src/cuda/cufinufft_plan_t.cu so the cufftDestroy call lives
// in exactly one TU; the same TU also performs the DeviceSwitcher
// dance so destruction is correct on multi-device systems.
class cufft_plan {
public:
  cufft_plan() = default;
  explicit cufft_plan(int device_id) noexcept : device_id_(device_id) {}
  cufft_plan(const cufft_plan &)            = delete;
  cufft_plan &operator=(const cufft_plan &) = delete;
  ~cufft_plan();

  cufftHandle get() const noexcept { return handle_; }
  cufftHandle *for_creation() noexcept { return &handle_; }
  explicit operator bool() const noexcept { return handle_ != 0; }
  void set_device_id(int id) noexcept { device_id_ = id; }

private:
  cufftHandle handle_ = 0;
  int device_id_      = 0;
};

// Forward declarations for friend templates.
template<typename T> struct cufinufft_plan_t;
template<typename T> struct cufinufft_gpu_data;

namespace cufinufft {
namespace spreadinterp {
template<typename T, int ndim, int ns>
void cuspread_nupts_driven(const cufinufft_plan_t<T> &, const cuda_complex<T> *,
                           cuda_complex<T> *, int);
template<typename T, int ndim, int ns>
void cuspread_subprob(const cufinufft_plan_t<T> &, const cuda_complex<T> *,
                      cuda_complex<T> *, int);
template<typename T, int ndim, int ns>
void cuspread_output_driven(const cufinufft_plan_t<T> &, const cuda_complex<T> *,
                            cuda_complex<T> *, int);
template<typename T, int ndim, int ns>
void cuspread3d_blockgather(const cufinufft_plan_t<T> &, const cuda_complex<T> *,
                            cuda_complex<T> *, int);
template<typename T, int ndim, int ns>
void cuinterp_nuptsdriven(const cufinufft_plan_t<T> &, cuda_complex<T> *,
                          const cuda_complex<T> *, int);
template<typename T, int ndim, int ns>
void cuinterp_subprob(const cufinufft_plan_t<T> &, cuda_complex<T> *,
                      const cuda_complex<T> *, int);

template<typename T, int Ndim>
void do_spread_nupts_driven(const cufinufft_plan_t<T> &, const cuda_complex<T> *,
                            cuda_complex<T> *, int);
template<typename T, int Ndim>
void do_spread_subprob(const cufinufft_plan_t<T> &, const cuda_complex<T> *,
                       cuda_complex<T> *, int);
template<typename T, int Ndim>
void do_spread_output_driven(const cufinufft_plan_t<T> &, const cuda_complex<T> *,
                             cuda_complex<T> *, int);
template<typename T>
void do_spread_blockgather_3d(const cufinufft_plan_t<T> &, const cuda_complex<T> *,
                              cuda_complex<T> *, int);

template<typename T, int Ndim> void do_prep_nupts_driven(cufinufft_plan_t<T> &);
template<typename T, int Ndim> void do_prep_subprob_and_OD(cufinufft_plan_t<T> &);
template<typename T> void do_prep_blockgather_3d(cufinufft_plan_t<T> &);

template<typename T, int Ndim>
void do_interp_nupts_driven(const cufinufft_plan_t<T> &, cuda_complex<T> *,
                            const cuda_complex<T> *, int);
template<typename T, int Ndim>
void do_interp_subprob(const cufinufft_plan_t<T> &, cuda_complex<T> *,
                       const cuda_complex<T> *, int);
} // namespace spreadinterp

namespace common {
template<typename T, typename V>
void cufinufft_set_shared_memory(V *, const cufinufft_plan_t<T> &);
} // namespace common
} // namespace cufinufft

template<typename T> inline T *dethrust(thrust::device_ptr<T> ptr) {
  return thrust::raw_pointer_cast(ptr);
}
template<typename T> inline thrust::device_ptr<T> enthrust(T *ptr) {
  return thrust::device_pointer_cast(ptr);
}

class DeviceSwitcher {
private:
  int orig_device;

  static int get_orig_device() noexcept {
    int device{};
    cudaGetDevice(&device);
    return device;
  }

public:
  explicit DeviceSwitcher(int newDevice) : orig_device{get_orig_device()} {
    if (cudaSetDevice(newDevice) != cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
  }

  ~DeviceSwitcher() {
    if (cudaSetDevice(orig_device) != cudaSuccess) {
      std::cerr << "failure reverting to original CUDA device; Exiting..." << std::endl;
      std::terminate();
    }
  }
};

template<typename T>
struct ThrustAllocatorAsync : public thrust::device_malloc_allocator<T> {
public:
  using Base      = thrust::device_malloc_allocator<T>;
  using pointer   = typename Base::pointer;
  using size_type = typename Base::size_type;

private:
  cudaStream_t stream;
  int deviceID;
  bool pool;

public:
  // Prefer explicit stream; no default ctor needed if you always pass alloc to
  // device_vector
  template<typename U> friend struct ThrustAllocatorAsync;

  explicit ThrustAllocatorAsync(cudaStream_t s, int ID, bool supports_pools)
      : stream(s), deviceID(ID), pool(supports_pools) {}

  template<typename U>
  ThrustAllocatorAsync(const ThrustAllocatorAsync<U> &o)
      : stream(o.stream), deviceID(o.deviceID), pool(o.pool) {}

  pointer allocate(size_type n) {
    DeviceSwitcher switcher(deviceID);
    T *p = nullptr;
    auto err =
        pool ? cudaMallocAsync(&p, n * sizeof(T), stream) : cudaMalloc(&p, n * sizeof(T));
    if (err != cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
    return enthrust(p);
  }

  void deallocate(pointer p, size_type) {
    DeviceSwitcher switcher(deviceID);
    auto err = pool ? cudaFreeAsync(dethrust(p), stream) : cudaFree(dethrust(p));
    if (err != cudaSuccess) { // something went really, really wrong, memory is corrupted
      std::cerr << "error while deallocating GPU memory! Exiting ..." << std::endl;
      std::terminate();
    }
  }
};

template<typename T> using gpu_array = thrust::device_vector<T, ThrustAllocatorAsync<T>>;

template<typename T> inline T *dethrust(gpu_array<T> &arr) {
  return thrust::raw_pointer_cast(arr.data());
}
template<typename T> inline const T *dethrust(const gpu_array<T> &arr) {
  return thrust::raw_pointer_cast(arr.data());
}
template<typename T>
inline cuda::std::array<T *, 3> dethrust(cuda::std::array<gpu_array<T>, 3> &arr) {
  cuda::std::array<T *, 3> res;
  for (int i = 0; i < 3; ++i) res[i] = dethrust(arr[i]);
  return res;
}
template<typename T>
inline cuda::std::array<const T *, 3> dethrust(
    const cuda::std::array<gpu_array<T>, 3> &arr) {
  cuda::std::array<const T *, 3> res;
  for (int i = 0; i < 3; ++i) res[i] = dethrust(arr[i]);
  return res;
}

template<typename T> struct cufinufft_plan_t {
  // -- Data members are private; host-side worker functions in
  //    cufinufft::spreadinterp / cufinufft::common access them through the
  //    friend declarations below. The kernel-side POD copy (cufinufft_gpu_data)
  //    is the boundary for device-side reads.

  cufinufft_plan_t() = delete;
  cufinufft_plan_t(int type_, int dim_, const int *nmodes, int iflag_, int ntransf_,
                   T tol_, const cufinufft_opts &opts_);
  cufinufft_plan_t &operator=(cufinufft_plan_t &) = delete;

  ~cufinufft_plan_t() = default;

  // Plan input config and a one-shot warning flag from the spreader setup
  // are public; tests, perftests and the C-API wrapper read them directly,
  // mirroring the FINUFFT_PLAN_T (CPU) convention of public `opts`.
  cufinufft_opts opts;
  finufft_spread_opts spopts;
  bool eps_too_small = false;

  // Dynamic shared-memory bytes required per kernel launch for spread/interp.
  // Public because per-method drivers (spreadinterp.hpp) and shared-memory
  // setup (common.hpp::cufinufft_set_shared_memory) call it from outside the
  // class. Wraps the free helper so callers don't re-thread plan members
  // (dim, nspread, gpu_binsize{x,y,z}, gpu_np) on every launch site.
  std::size_t shared_memory_required() const;

  void setpts(int nj, const T *d_kx, const T *d_ky, const T *d_kz, int nk, const T *d_s,
              const T *d_t, const T *d_u);
  // FIXME: we want to make this "const" in the future
  void execute(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const;

private:
  // Worker functions and the POD-copy helper need direct access to private
  // state (mutating prep helpers resize bin/subprob arrays; spread/interp
  // drivers read M, nf, totalnumsubprob, stream, kxyz, ...).
  template<typename> friend struct cufinufft_gpu_data;

  template<typename U, int N, int K>
  friend void cufinufft::spreadinterp::cuspread_nupts_driven(
      const cufinufft_plan_t<U> &, const cuda_complex<U> *, cuda_complex<U> *, int);
  template<typename U, int N, int K>
  friend void cufinufft::spreadinterp::cuspread_subprob(
      const cufinufft_plan_t<U> &, const cuda_complex<U> *, cuda_complex<U> *, int);
  template<typename U, int N, int K>
  friend void cufinufft::spreadinterp::cuspread_output_driven(
      const cufinufft_plan_t<U> &, const cuda_complex<U> *, cuda_complex<U> *, int);
  template<typename U, int N, int K>
  friend void cufinufft::spreadinterp::cuspread3d_blockgather(
      const cufinufft_plan_t<U> &, const cuda_complex<U> *, cuda_complex<U> *, int);
  template<typename U, int N, int K>
  friend void cufinufft::spreadinterp::cuinterp_nuptsdriven(
      const cufinufft_plan_t<U> &, cuda_complex<U> *, const cuda_complex<U> *, int);
  template<typename U, int N, int K>
  friend void cufinufft::spreadinterp::cuinterp_subprob(
      const cufinufft_plan_t<U> &, cuda_complex<U> *, const cuda_complex<U> *, int);

  template<typename U, int N>
  friend void cufinufft::spreadinterp::do_spread_nupts_driven(
      const cufinufft_plan_t<U> &, const cuda_complex<U> *, cuda_complex<U> *, int);
  template<typename U, int N>
  friend void cufinufft::spreadinterp::do_spread_subprob(
      const cufinufft_plan_t<U> &, const cuda_complex<U> *, cuda_complex<U> *, int);
  template<typename U, int N>
  friend void cufinufft::spreadinterp::do_spread_output_driven(
      const cufinufft_plan_t<U> &, const cuda_complex<U> *, cuda_complex<U> *, int);
  template<typename U>
  friend void cufinufft::spreadinterp::do_spread_blockgather_3d(
      const cufinufft_plan_t<U> &, const cuda_complex<U> *, cuda_complex<U> *, int);

  template<typename U, int N>
  friend void cufinufft::spreadinterp::do_prep_nupts_driven(cufinufft_plan_t<U> &);
  template<typename U, int N>
  friend void cufinufft::spreadinterp::do_prep_subprob_and_OD(cufinufft_plan_t<U> &);
  template<typename U>
  friend void cufinufft::spreadinterp::do_prep_blockgather_3d(cufinufft_plan_t<U> &);

  template<typename U, int N>
  friend void cufinufft::spreadinterp::do_interp_nupts_driven(
      const cufinufft_plan_t<U> &, cuda_complex<U> *, const cuda_complex<U> *, int);
  template<typename U, int N>
  friend void cufinufft::spreadinterp::do_interp_subprob(
      const cufinufft_plan_t<U> &, cuda_complex<U> *, const cuda_complex<U> *, int);

  template<typename U, typename V>
  friend void cufinufft::common::cufinufft_set_shared_memory(V *,
                                                             const cufinufft_plan_t<U> &);

  bool supports_pools = false;

  ThrustAllocatorAsync<std::byte> alloc{(cudaStream_t)opts.gpu_stream, opts.gpu_device_id,
                                        supports_pools};

  // Plan-config invariants — set in the ctor's member-initializer list and
  // never mutated thereafter.
  const int type;
  const int dim;
  const int ntransf;
  const int iflag;
  const T tol;

  CUFINUFFT_BIGINT M                          = 0;
  cuda::std::array<CUFINUFFT_BIGINT, 3> nf123 = {0, 0, 0};
  cuda::std::array<CUFINUFFT_BIGINT, 3> mstu  = {0, 0, 0};
  int batchsize                               = 0;

  int totalnumsubprob                         = 0;
  cuda::std::array<gpu_array<T>, 3> fwkerhalf = {
      gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}};

  // for type 1,2 it is a pointer to kx, ky, kz (no new allocs), for type 3 it
  // for t3: allocated as "primed" (scaled) src pts x'_j, etc
  cuda::std::array<const T *, 3> kxyz     = {nullptr, nullptr, nullptr};
  cuda::std::array<gpu_array<T>, 3> kxyzp = {
      gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}};

  // Type 3 specific
  struct {
    cuda::std::array<T, 3> X = {0, 0, 0}, C = {0, 0, 0}, S = {0, 0, 0}, D = {0, 0, 0},
                           h = {0, 0, 0}, gam = {0, 0, 0};
  } type3_params;
  int N                                  = 0; // number of NU freq pts (type 3 only)
  CUFINUFFT_BIGINT nf                    = 0;
  cuda::std::array<const T *, 3> STU     = {nullptr, nullptr, nullptr};
  cuda::std::array<gpu_array<T>, 3> STUp = {
      gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}};
  // inner type 2 plan for type 3
  std::unique_ptr<cufinufft_plan_t<T>> t2_plan;

  gpu_array<cuda_complex<T>> prephase{0, alloc}; // pre-phase, for all input NU pts
  gpu_array<cuda_complex<T>> deconv{0, alloc};   // reciprocal of kernel FT, phase, all
                                                 // output NU pts

  // Arrays that used in subprob method
  gpu_array<int> idxnupts{0, alloc};   // length: #nupts, index of the nupts in the
                                       // bin-sorted order
  gpu_array<int> sortidx{0, alloc};    // length: #nupts, order inside the bin the nupt
                                       // belongs to
  gpu_array<int> numsubprob{0, alloc}; // length: #bins,  number of subproblems in each
                                       // bin
  gpu_array<int> binsize{0, alloc}; // length: #bins, number of nonuniform ponits in each
                                    // bin
  gpu_array<int> binstartpts{0, alloc}; // length: #bins, exclusive scan of array binsize
  gpu_array<int> subprob_to_bin{0, alloc}; // length: #subproblems, the bin the subproblem
                                           // works on
  gpu_array<int> subprobstartpts{0, alloc}; // length: #bins, exclusive scan of array
                                            // numsubprob

  // Arrays for 3d (need to sort out)
  gpu_array<int> numnupts{0, alloc};
  gpu_array<int> subprob_to_nupts{0, alloc};

  cufft_plan fftplan;
  cudaStream_t stream = 0;

  void exec1(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const;
  // The "ntransf_override" parameter is only needed when a type 3 plan calls
  // its inner type 2 plan. Leave at default in all other circumstances!
  void exec2(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
             std::optional<int> ntransf_override = std::optional<int>()) const;
  void exec3(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const;

  void deconvolve(cuda_complex<T> *fw, cuda_complex<T> *fk, int blksize) const;
  template<int modeord, int ndim>
  void deconvolve_nd(cuda_complex<T> *fw, cuda_complex<T> *fk, int blksize) const;

  void setpts_12(int M_, const T *d_kx, const T *d_ky, const T *d_kz);
  void allocate();
  void allocate_nupts();

  // Helpers migrated from free functions in cufinufft::common / cufinufft::utils.
  // Use this->opts and this->spopts; called only from plan setup.
  void set_nf_type12(CUFINUFFT_BIGINT ms, CUFINUFFT_BIGINT *nf, CUFINUFFT_BIGINT b) const;
  std::tuple<CUFINUFFT_BIGINT, T, T> set_nhg_type3(T S, T X) const;
  void onedim_fseries_kernel_precomp(CUFINUFFT_BIGINT nf, T *f, T *phase) const;

  // Spread/interp drivers — implementations live in src/cuda/spreadinterp.cu.
  // prep_spreadinterp() is mutating: it sets up the bin-sort / subproblem /
  // block-gather state in setpts(), and that state is consumed by both
  // spread() and interp().
  void prep_spreadinterp();
  void spread(const cuda_complex<T> *c, cuda_complex<T> *fw, int blksize) const;
  void interp(cuda_complex<T> *c, const cuda_complex<T> *fw, int blksize) const;

  // Per-method spread/interp entry points. Bodies live in per-method TUs
  // (src/cuda/spread_*.cu, src/cuda/interp_*.cu) to preserve compile
  // parallelism across nvcc invocations.
  void prep_nupts_driven();   // gpu_method = 1
  void prep_subprob_and_OD(); // gpu_method = 2 or 3
  void prep_blockgather_3d(); // gpu_method = 4 (3D-only)

  void spread_nupts_driven(const cuda_complex<T> *c, cuda_complex<T> *fw,
                           int blksize) const;
  void spread_subprob(const cuda_complex<T> *c, cuda_complex<T> *fw, int blksize) const;
  void spread_output_driven(const cuda_complex<T> *c, cuda_complex<T> *fw,
                            int blksize) const;
  void spread_blockgather_3d(const cuda_complex<T> *c, cuda_complex<T> *fw,
                             int blksize) const;

  void interp_nupts_driven(cuda_complex<T> *c, const cuda_complex<T> *fw,
                           int blksize) const;
  void interp_subprob(cuda_complex<T> *c, const cuda_complex<T> *fw, int blksize) const;
};

// This class contains a subset of the information stored in
// cufinufft_plan_t, in a shape that can be copied to GPU.
// Its only intended use is for (implicit) on-the-fly construction
// from a full cufinufft_plan_t object when launching a kernel, such
// that the kernel has convenient access to the relevant parameters
// and arrays. Do not cache such objects or its members!
template<typename T> struct cufinufft_gpu_data {
  cufinufft_opts opts;
  finufft_spread_opts spopts;

  // Derived ES-kernel parameters, cached so spread/interp kernels do not
  // recompute them on every launch. es_c == (2/ns)^2; sigma == upsampfac.
  T sigma   = 0;
  T es_c    = 0;
  T es_beta = 0;

  int type                                    = 0;
  int dim                                     = 0;
  CUFINUFFT_BIGINT M                          = 0;
  cuda::std::array<CUFINUFFT_BIGINT, 3> nf123 = {0, 0, 0};
  cuda::std::array<CUFINUFFT_BIGINT, 3> mstu  = {0, 0, 0};
  int ntransf                                 = 0;
  int batchsize                               = 0;
  int iflag                                   = 0;

  int totalnumsubprob = 0;

  // for type 1,2 it is a pointer to kx, ky, kz (no new allocs), for type 3 it
  // for t3: allocated as "primed" (scaled) src pts x'_j, etc
  cuda::std::array<const T *, 3> xyz = {nullptr, nullptr, nullptr};

  int N                              = 0; // number of NU freq pts (type 3 only)
  CUFINUFFT_BIGINT nf                = 0;
  cuda::std::array<const T *, 3> STU = {nullptr, nullptr, nullptr};
  T tol                              = 0;

  const cuda_complex<T> *prephase = nullptr; // pre-phase, for all input NU pts
  const cuda_complex<T> *deconv   = nullptr; // reciprocal of kernel FT, phase, all
                                             // output NU pts

  // Arrays that used in subprob method
  const int *idxnupts = nullptr;   // length: #nupts, index of the nupts in the
                                   // bin-sorted order
  const int *sortidx = nullptr;    // length: #nupts, order inside the bin the nupt
                                   // belongs to
  const int *numsubprob = nullptr; // length: #bins,  number of subproblems in each
                                   // bin
  const int *binsize = nullptr;    // length: #bins, number of nonuniform ponits in each
                                   // bin
  const int *binstartpts    = nullptr;  // length: #bins, exclusive scan of array binsize
  const int *subprob_to_bin = nullptr;  // length: #subproblems, the bin the subproblem
                                        // works on
  const int *subprobstartpts = nullptr; // length: #bins, exclusive scan of array
                                        // numsubprob

  // Arrays for 3d (need to sort out)
  const int *numnupts         = nullptr;
  const int *subprob_to_nupts = nullptr;

  cufinufft_gpu_data() = delete;
  cufinufft_gpu_data(const cufinufft_plan_t<T> &orig)
      : opts(orig.opts), spopts(orig.spopts), sigma(orig.spopts.upsampfac),
        es_c(T(4) / T(orig.spopts.nspread * orig.spopts.nspread)),
        es_beta(orig.spopts.beta), type(orig.type), dim(orig.dim), M(orig.M),
        nf123(orig.nf123), mstu(orig.mstu), ntransf(orig.ntransf),
        batchsize(orig.batchsize), iflag(orig.iflag),
        totalnumsubprob(orig.totalnumsubprob), xyz(orig.kxyz), N(orig.N), nf(orig.nf),
        STU(orig.STU), tol(orig.tol), prephase(dethrust(orig.prephase)),
        deconv(dethrust(orig.deconv)), idxnupts(dethrust(orig.idxnupts)),
        sortidx(dethrust(orig.sortidx)), numsubprob(dethrust(orig.numsubprob)),
        binsize(dethrust(orig.binsize)), binstartpts(dethrust(orig.binstartpts)),
        subprob_to_bin(dethrust(orig.subprob_to_bin)),
        subprobstartpts(dethrust(orig.subprobstartpts)),
        numnupts(dethrust(orig.numnupts)),
        subprob_to_nupts(dethrust(orig.subprob_to_nupts)) {}
};

#endif
