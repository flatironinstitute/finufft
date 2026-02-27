#ifndef CUFINUFFT_TYPES_H
#define CUFINUFFT_TYPES_H

#include <cufft.h>
#include <cuda/std/array>

#include <cufinufft/defs.h>
#include <cufinufft_opts.h>
#include <finufft_common/spread_opts.h>
#include <type_traits>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>

#include <cuComplex.h>

using CUFINUFFT_BIGINT=int;

// Marco Barbone 8/5/2024, replaced the ugly trick with std::conditional
// to define cuda_complex
// by using std::conditional and std::is_same, we can define cuda_complex
// if T is float, cuda_complex<T> is cuFloatComplex
// if T is double, cuda_complex<T> is cuDoubleComplex
// where cuFloatComplex and cuDoubleComplex are defined in cuComplex.h
// TODO: migrate to cuda/std/complex and remove this
//       Issue: cufft seems not to support cuda::std::complex
//       A reinterpret_cast should be enough
template<typename T>
using cuda_complex = typename std::conditional<
    std::is_same<T, float>::value, cuFloatComplex,
    typename std::conditional<std::is_same<T, double>::value, cuDoubleComplex,
                              void>::type>::type;

template<typename T> inline T* dethrust(thrust::device_ptr<T> ptr) {
  return thrust::raw_pointer_cast(ptr);
  }
template<typename T> inline thrust::device_ptr<T> enthrust(T *ptr) {
  return thrust::device_pointer_cast(ptr);
  }

template <typename T>
struct ThrustAllocatorAsync : public thrust::device_malloc_allocator<T> {
  public:
    using Base      = thrust::device_malloc_allocator<T>;
    using pointer   = typename Base::pointer;
    using size_type = typename Base::size_type;

  private:
    cudaStream_t stream;
    bool pool;

  public:
    // We need a default constructor as long as we do not fully C++ify
    // the NUFFT plan struct.
    ThrustAllocatorAsync() : stream(0), pool (false) {}
    // Prefer explicit stream; no default ctor needed if you always pass alloc to device_vector
    explicit ThrustAllocatorAsync(cudaStream_t s, bool supports_pools) : stream(s), pool (supports_pools) {}

    pointer allocate(size_type n) {
      T* p = nullptr;
      auto err = pool ? cudaMallocAsync(&p, n*sizeof(T), stream)
                      : cudaMalloc(&p, n*sizeof(T));
      if (err!=cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
      return enthrust(p);
    }
  
    void deallocate(pointer p, size_type) {
      auto err = pool ? cudaFreeAsync(dethrust(p), stream)
                      : cudaFree(dethrust(p));
      if (err!=cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
    }
};

template<typename T> using gpuArray = thrust::device_vector<T, ThrustAllocatorAsync<T>>;

template<typename T> inline T* dethrust(gpuArray<T> &arr) {
  return thrust::raw_pointer_cast(arr.data());
  }
template<typename T> inline cuda::std::array<T *,3> dethrust(cuda::std::array<gpuArray<T>,3> &arr) {
  cuda::std::array<T *,3> res;
  for (int i=0; i<3; ++i) res[i] = dethrust(arr[i]);
  return res;
  }

template<typename T> struct cufinufft_plan_t {
  cufinufft_opts opts;
  bool supports_pools=false;
  finufft_spread_opts spopts;

  ThrustAllocatorAsync<int> ialloc{(cudaStream_t)opts.gpu_stream, supports_pools};
  ThrustAllocatorAsync<T> alloc{(cudaStream_t)opts.gpu_stream, supports_pools};
  ThrustAllocatorAsync<cuda_complex<T>> calloc{(cudaStream_t)opts.gpu_stream, supports_pools};

  int type=0;
  int dim=0;
  CUFINUFFT_BIGINT M=0;
  cuda::std::array<CUFINUFFT_BIGINT,3> nf123={0,0,0};
  cuda::std::array<CUFINUFFT_BIGINT,3> mstu={0,0,0};
  int ntransf=0;
  int batchsize=0;
  int iflag=0;

  int totalnumsubprob=0;
  cuda::std::array<gpuArray<T>,3> fwkerhalf={gpuArray<T>{0, alloc},gpuArray<T>{0,alloc},gpuArray<T>{0,alloc}};

  // for type 1,2 it is a pointer to kx, ky, kz (no new allocs), for type 3 it
  // for t3: allocated as "primed" (scaled) src pts x'_j, etc
  cuda::std::array<T *,3> kxyz={nullptr,nullptr,nullptr};
  cuda::std::array<gpuArray<T>,3> kxyzp={gpuArray<T>{0,alloc},gpuArray<T>{0,alloc},gpuArray<T>{0,alloc}};
  gpuArray<cuda_complex<T>> CpBatch{0, calloc}; // working array of prephased strengths

  // no allocs here
  cuda_complex<T> *c=nullptr;
  gpuArray<cuda_complex<T>> fwp{0,calloc};
  cuda_complex<T> *fw=nullptr;
  cuda_complex<T> *fk=nullptr;

  // Type 3 specific
  struct {
    cuda::std::array<T,3> X, C, S, D, h, gam;
  } type3_params;
  int N=0;                        // number of NU freq pts (type 3 only)
  CUFINUFFT_BIGINT nf=0;
  cuda::std::array<T *,3> STU={nullptr,nullptr,nullptr};
  cuda::std::array<gpuArray<T>,3> STUp={gpuArray<T>{0, alloc},gpuArray<T>{0,alloc},gpuArray<T>{0,alloc}};
  T tol=0;
  // inner type 2 plan for type 3
  cufinufft_plan_t<T> *t2_plan=nullptr;
  // new allocs.
  // FIXME: convert to device vectors to use resize
  gpuArray<cuda_complex<T>> prephase{0, calloc}; // pre-phase, for all input NU pts
  gpuArray<cuda_complex<T>> deconv{0,calloc};   // reciprocal of kernel FT, phase, all output NU pts

  // Arrays that used in subprob method
  gpuArray<int> idxnupts{0,ialloc};        // length: #nupts, index of the nupts in the bin-sorted order
  gpuArray<int> sortidx{0,ialloc};         // length: #nupts, order inside the bin the nupt belongs to
  gpuArray<int> numsubprob{0,ialloc};      // length: #bins,  number of subproblems in each bin
  gpuArray<int> binsize{0,ialloc};         // length: #bins, number of nonuniform ponits in each bin
  gpuArray<int> binstartpts{0,ialloc};     // length: #bins, exclusive scan of array binsize
  gpuArray<int> subprob_to_bin{0,ialloc};  // length: #subproblems, the bin the subproblem works on
  gpuArray<int> subprobstartpts{0,ialloc}; // length: #bins, exclusive scan of array numsubprob

  // Arrays for 3d (need to sort out)
  gpuArray<int> numnupts{0,ialloc};
  gpuArray<int> subprob_to_nupts{0,ialloc};

  cufftHandle fftplan=0;
  cudaStream_t stream=0;

  cufinufft_plan_t(const cufinufft_opts &opts_, bool supports_pools_)
    : opts(opts_), supports_pools(supports_pools_) {}
};

template<typename T> static inline constexpr cufftType_t cufft_type();
template<> inline constexpr cufftType_t cufft_type<float>() { return CUFFT_C2C; }

template<> inline constexpr cufftType_t cufft_type<double>() { return CUFFT_Z2Z; }

static inline cufftResult cufft_ex(cufftHandle plan, cufftComplex *idata,
                                   cufftComplex *odata, int direction) {
  return cufftExecC2C(plan, idata, odata, direction);
}
static inline cufftResult cufft_ex(cufftHandle plan, cufftDoubleComplex *idata,
                                   cufftDoubleComplex *odata, int direction) {
  return cufftExecZ2Z(plan, idata, odata, direction);
}

#endif
