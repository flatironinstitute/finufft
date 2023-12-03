#ifndef _SCTL_FMM_WRAPPER_HPP_
#define _SCTL_FMM_WRAPPER_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)

#include <map>
#include <string>

#ifdef SCTL_HAVE_PVFMM
namespace pvfmm {
  template <class Real> class Kernel;
  template <class Real> class MPI_Node;
  template <class Node> class FMM_Node;
  template <class FMM_Node> class FMM_Pts;
  template <class FMM_Mat> class FMM_Tree;
  template <class Real> using PtFMM_Node = FMM_Node<MPI_Node<Real>>;
  template <class Real> using PtFMM      = FMM_Pts<PtFMM_Node<Real>>;
  template <class Real> using PtFMM_Tree = FMM_Tree<PtFMM<Real>>;
}
#endif

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;

template <class Real, Integer DIM> class ParticleFMM {
  public:

    ParticleFMM(const ParticleFMM&) = delete;
    ParticleFMM& operator= (const ParticleFMM&) = delete;

    ParticleFMM(const Comm& comm = Comm::Self());

    ~ParticleFMM();

    void SetComm(const Comm& comm);

    void SetAccuracy(Integer digits);

    template <class KerM2M, class KerM2L, class KerL2L> void SetKernels(const KerM2M& ker_m2m, const KerM2L& ker_m2l, const KerL2L& ker_l2l);
    template <class KerS2M, class KerS2L> void AddSrc(const std::string& name, const KerS2M& ker_s2m, const KerS2L& ker_s2l);
    template <class KerM2T, class KerL2T> void AddTrg(const std::string& name, const KerM2T& ker_m2t, const KerL2T& ker_l2t);
    template <class KerS2T> void SetKernelS2T(const std::string& src_name, const std::string& trg_name, const KerS2T& ker_s2t);

    void DeleteSrc(const std::string& name);
    void DeleteTrg(const std::string& name);

    void SetSrcCoord(const std::string& name, const Vector<Real>& src_coord, const Vector<Real>& src_normal = Vector<Real>());
    void SetSrcDensity(const std::string& name, const Vector<Real>& src_density);
    void SetTrgCoord(const std::string& name, const Vector<Real>& trg_coord);

    void Eval(Vector<Real>& U, const std::string& trg_name) const;
    void EvalDirect(Vector<Real>& U, const std::string& trg_name) const;

    static void test(const Comm& comm);

  private:

    struct FMMKernels {
      Iterator<char> ker_m2m, ker_m2l, ker_l2l;
      Integer dim_mul_ch, dim_mul_eq;
      Integer dim_loc_ch, dim_loc_eq;

      void (*ker_m2m_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_m2l_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_l2l_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

      void (*delete_ker_m2m)(Iterator<char> ker);
      void (*delete_ker_m2l)(Iterator<char> ker);
      void (*delete_ker_l2l)(Iterator<char> ker);

      #ifdef SCTL_HAVE_PVFMM
      pvfmm::Kernel<Real> pvfmm_ker_m2m;
      pvfmm::Kernel<Real> pvfmm_ker_m2l;
      pvfmm::Kernel<Real> pvfmm_ker_l2l;
      #endif
    };
    struct SrcData {
      Vector<Real> X, Xn, F;
      Iterator<char> ker_s2m, ker_s2l;
      Integer dim_src, dim_mul_ch, dim_loc_ch, dim_normal;

      void (*ker_s2m_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_s2l_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

      void (*delete_ker_s2m)(Iterator<char> ker);
      void (*delete_ker_s2l)(Iterator<char> ker);

      #ifdef SCTL_HAVE_PVFMM
      pvfmm::Kernel<Real> pvfmm_ker_s2m;
      pvfmm::Kernel<Real> pvfmm_ker_s2l;
      StaticArray<Real, DIM*2> bbox;
      #endif
    };
    struct TrgData {
      Vector<Real> X, U;
      Iterator<char> ker_m2t, ker_l2t;
      Integer dim_mul_eq, dim_loc_eq, dim_trg;

      void (*ker_m2t_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_l2t_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

      void (*delete_ker_m2t)(Iterator<char> ker);
      void (*delete_ker_l2t)(Iterator<char> ker);

      #ifdef SCTL_HAVE_PVFMM
      pvfmm::Kernel<Real> pvfmm_ker_m2t;
      pvfmm::Kernel<Real> pvfmm_ker_l2t;
      StaticArray<Real, DIM*2> bbox;
      #endif
    };
    struct S2TData {
      Iterator<char> ker_s2t;
      Integer dim_src, dim_trg, dim_normal;

      void (*ker_s2t_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_s2t_eval_omp)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

      void (*delete_ker_s2t)(Iterator<char> ker);

      #ifdef SCTL_HAVE_PVFMM
      mutable Real bbox_scale;
      mutable StaticArray<Real,DIM> bbox_offset;
      mutable Vector<Real> src_scal_exp, trg_scal_exp;
      mutable Vector<Real> src_scal, trg_scal;
      mutable pvfmm::Kernel<Real> pvfmm_ker_s2t;
      mutable pvfmm::PtFMM_Tree<Real>* tree_ptr;
      mutable pvfmm::PtFMM<Real> fmm_ctx;
      mutable bool setup_tree;
      mutable bool setup_ker;
      #endif
    };

    static void BuildSrcTrgScal(const S2TData& s2t_data, bool verbose);

    template <class Ker> static void DeleteKer(Iterator<char> ker);

    void CheckKernelDims() const;

    void DeleteS2T(const std::string& src_name, const std::string& trg_name);

    #ifdef SCTL_HAVE_PVFMM
    template <class SCTLKernel, bool use_dummy_normal=false> struct PVFMMKernelFn; // construct PVFMMKernel from SCTLKernel

    void EvalPVFMM(Vector<Real>& U, const std::string& trg_name) const;
    #endif

    FMMKernels fmm_ker;
    std::map<std::string, SrcData> src_map;
    std::map<std::string, TrgData> trg_map;
    std::map<std::pair<std::string,std::string>, S2TData> s2t_map;

    Comm comm_;
    Integer digits_;
};

}  // end namespace

#include SCTL_INCLUDE(fmm-wrapper.txx)

#endif  //_SCTL_FMM_WRAPPER_HPP_
