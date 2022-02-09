#include SCTL_INCLUDE(kernel_functions.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(matrix.hpp)

#ifdef SCTL_HAVE_PVFMM
#include <pvfmm.hpp>
#endif

namespace SCTL_NAMESPACE {

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::test(const Comm& comm) {
  if (DIM != 3) return ParticleFMM<Real,3>::test(comm);

  Stokes3D_FSxU kernel_m2l;
  Stokes3D_FxU kernel_sl;
  Stokes3D_DxU kernel_dl;
  srand48(comm.Rank());

  // Create target and source vectors.
  const Long N = 50000/comm.Size();
  Vector<Real> trg_coord(N*DIM);
  Vector<Real>  sl_coord(N*DIM);
  Vector<Real>  dl_coord(N*DIM);
  Vector<Real>  dl_norml(N*DIM);
  for (auto& a : trg_coord) a = (Real)(drand48()-0.5);
  for (auto& a :  sl_coord) a = (Real)(drand48()-0.5);
  for (auto& a :  dl_coord) a = (Real)(drand48()-0.5);
  for (auto& a :  dl_norml) a = (Real)(drand48()-0.5);
  Long n_sl  =  sl_coord.Dim()/DIM;
  Long n_dl  =  dl_coord.Dim()/DIM;

  // Set source charges.
  Vector<Real> sl_den(n_sl*kernel_sl.SrcDim());
  Vector<Real> dl_den(n_dl*kernel_dl.SrcDim());
  for (auto& a : sl_den) a = (Real)(drand48() - 0.5);
  for (auto& a : dl_den) a = (Real)(drand48() - 0.5);

  ParticleFMM fmm(comm);
  fmm.SetAccuracy(10);
  fmm.SetKernels(kernel_m2l, kernel_m2l, kernel_sl);
  fmm.AddTrg("Potential", kernel_m2l, kernel_sl);
  fmm.AddSrc("SingleLayer", kernel_sl, kernel_sl);
  fmm.AddSrc("DoubleLayer", kernel_dl, kernel_dl);
  fmm.SetKernelS2T("SingleLayer", "Potential",kernel_sl);
  fmm.SetKernelS2T("DoubleLayer", "Potential",kernel_dl);

  fmm.SetTrgCoord("Potential", trg_coord);
  fmm.SetSrcCoord("SingleLayer", sl_coord);
  fmm.SetSrcCoord("DoubleLayer", dl_coord, dl_norml);

  fmm.SetSrcDensity("SingleLayer", sl_den);
  fmm.SetSrcDensity("DoubleLayer", dl_den);

  Vector<Real> Ufmm, Uref;
  fmm.Eval(Ufmm, "Potential"); // Warm-up run
  Ufmm = 0;

  Profile::Enable(true);
  Profile::Tic("FMM-Eval", &comm);
  fmm.Eval(Ufmm, "Potential");
  Profile::Toc();
  Profile::Tic("Direct", &comm);
  fmm.EvalDirect(Uref, "Potential");
  Profile::Toc();
  Profile::print(&comm);

  Vector<Real> Uerr = Uref - Ufmm;
  { // Print error
    StaticArray<Real,2> loc_err{0,0}, glb_err{0,0};
    for (const auto& a : Uerr) loc_err[0] = std::max<Real>(loc_err[0], fabs(a));
    for (const auto& a : Uref) loc_err[1] = std::max<Real>(loc_err[1], fabs(a));
    comm.Allreduce<Real>(loc_err, glb_err, 2, Comm::CommOp::MAX);
    if (!comm.Rank()) std::cout<<"Maximum relative error: "<<glb_err[0]/glb_err[1]<<'\n';
  }
}

template <Integer DIM, class Real> void BoundingBox(StaticArray<Real,DIM*2>& bbox, const Vector<Real>& X, const Comm& comm) {
  const Long N = X.Dim()/DIM;
  SCTL_ASSERT(X.Dim() == N * DIM);

  StaticArray<Real, DIM> bbox0, bbox1;
  { // Init bbox0, bbox1
    for (Integer k = 0; k < DIM; k++) {
      bbox0[k] = bbox1[k] = (N ? X[k] : (Real)0);
    }

    // Handle (N == 0) case
    StaticArray<Real, DIM> bbox0_glb, bbox1_glb;
    comm.Allreduce((ConstIterator<Real>)bbox0, (Iterator<Real>)bbox0_glb, DIM, Comm::CommOp::MIN);
    comm.Allreduce((ConstIterator<Real>)bbox1, (Iterator<Real>)bbox1_glb, DIM, Comm::CommOp::MAX);
    for (Integer k = 0; k < DIM; k++) {
      bbox0[k] = bbox1[k] = (fabs(bbox0_glb[k]) > fabs(bbox1_glb[k]) ? bbox0_glb[k] : bbox1_glb[k]);
    }
  }
  for (Long i = 0; i < N; i++) {
    for (Integer k = 0; k < DIM; k++) {
      bbox0[k] = std::min<Real>(bbox0[k], X[i*DIM+k]);
      bbox1[k] = std::max<Real>(bbox1[k], X[i*DIM+k]);
    }
  }

  StaticArray<Real, DIM> bbox0_glb, bbox1_glb;
  comm.Allreduce((ConstIterator<Real>)bbox0, (Iterator<Real>)bbox0_glb, DIM, Comm::CommOp::MIN);
  comm.Allreduce((ConstIterator<Real>)bbox1, (Iterator<Real>)bbox1_glb, DIM, Comm::CommOp::MAX);
  for (Integer k = 0; k < DIM; k++) {
    bbox[k*2+0] = bbox0_glb[k];
    bbox[k*2+1] = bbox1_glb[k];
  }
}

template <class Real, Integer DIM> ParticleFMM<Real,DIM>::ParticleFMM(const Comm& comm) : comm_(comm), digits_(10) {
  fmm_ker.ker_m2m = NullIterator<char>();
  fmm_ker.ker_m2l = NullIterator<char>();
  fmm_ker.ker_l2l = NullIterator<char>();
}

template <class Real, Integer DIM> ParticleFMM<Real,DIM>::~ParticleFMM() {
  Vector<std::string> src_lst, trg_lst;
  for (auto& it : src_map) src_lst.PushBack(it.first);
  for (auto& it : trg_map) trg_lst.PushBack(it.first);
  for (const auto& name : src_lst) DeleteSrc(name);
  for (const auto& name : trg_lst) DeleteTrg(name);

  Vector<std::pair<std::string,std::string>> s2t_lst;
  for (auto& it : s2t_map) s2t_lst.PushBack(it.first);
  for (const auto& name : s2t_lst) DeleteS2T(name.first, name.second);

  if (fmm_ker.ker_m2m != NullIterator<char>()) fmm_ker.delete_ker_m2m(fmm_ker.ker_m2m);
  if (fmm_ker.ker_m2l != NullIterator<char>()) fmm_ker.delete_ker_m2l(fmm_ker.ker_m2l);
  if (fmm_ker.ker_l2l != NullIterator<char>()) fmm_ker.delete_ker_l2l(fmm_ker.ker_l2l);
}

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::SetComm(const Comm& comm) {
  comm_ = comm;
  #ifdef SCTL_HAVE_PVFMM
  if (DIM == 3) {
    for (auto& it : s2t_map) {
      it.second.setup_ker = true;
      it.second.setup_tree = true;
    }
  }
  #endif
}

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::SetAccuracy(Integer digits) {
  #ifdef SCTL_HAVE_PVFMM
  if (DIM == 3 && digits != digits_) {
    for (auto& it : s2t_map) {
      it.second.setup_ker = true;
      it.second.setup_tree = true;
    }
  }
  #endif
  digits_ = digits;
}

template <class Real, Integer DIM> template <class KerM2M, class KerM2L, class KerL2L> void ParticleFMM<Real,DIM>::SetKernels(const KerM2M& ker_m2m, const KerM2L& ker_m2l, const KerL2L& ker_l2l) {
  if (fmm_ker.ker_m2m != NullIterator<char>()) fmm_ker.delete_ker_m2m(fmm_ker.ker_m2m);
  if (fmm_ker.ker_m2l != NullIterator<char>()) fmm_ker.delete_ker_m2l(fmm_ker.ker_m2l);
  if (fmm_ker.ker_l2l != NullIterator<char>()) fmm_ker.delete_ker_l2l(fmm_ker.ker_l2l);

  fmm_ker.ker_m2m = (Iterator<char>)aligned_new<KerM2M>(1);
  fmm_ker.ker_m2l = (Iterator<char>)aligned_new<KerM2L>(1);
  fmm_ker.ker_l2l = (Iterator<char>)aligned_new<KerL2L>(1);
  (*(Iterator<KerM2M>)fmm_ker.ker_m2m) = ker_m2m;
  (*(Iterator<KerM2L>)fmm_ker.ker_m2l) = ker_m2l;
  (*(Iterator<KerL2L>)fmm_ker.ker_l2l) = ker_l2l;

  fmm_ker.dim_mul_eq = ker_m2m.SrcDim();
  fmm_ker.dim_mul_ch = ker_m2m.TrgDim();
  fmm_ker.dim_loc_eq = ker_l2l.SrcDim();
  fmm_ker.dim_loc_ch = ker_l2l.TrgDim();
  SCTL_ASSERT(ker_m2m.CoordDim() == DIM);
  SCTL_ASSERT(ker_m2l.CoordDim() == DIM);
  SCTL_ASSERT(ker_l2l.CoordDim() == DIM);
  SCTL_ASSERT(ker_m2l.SrcDim() == fmm_ker.dim_mul_eq);
  SCTL_ASSERT(ker_m2l.TrgDim() == fmm_ker.dim_loc_ch);

  fmm_ker.ker_m2m_eval = KerM2M::template Eval<Real,false>;
  fmm_ker.ker_m2l_eval = KerM2L::template Eval<Real,false>;
  fmm_ker.ker_l2l_eval = KerL2L::template Eval<Real,false>;

  fmm_ker.delete_ker_m2m = DeleteKer<KerM2M>;
  fmm_ker.delete_ker_m2l = DeleteKer<KerM2L>;
  fmm_ker.delete_ker_l2l = DeleteKer<KerL2L>;

  #ifdef SCTL_HAVE_PVFMM
  if (DIM == 3) {
    fmm_ker.pvfmm_ker_m2m = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerM2M>::template Eval<Real>>(ker_m2m.Name().c_str(), DIM, std::pair<int,int>(ker_m2m.SrcDim(), ker_m2m.TrgDim()));
    fmm_ker.pvfmm_ker_m2l = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerM2L>::template Eval<Real>>(ker_m2l.Name().c_str(), DIM, std::pair<int,int>(ker_m2l.SrcDim(), ker_m2l.TrgDim()));
    fmm_ker.pvfmm_ker_l2l = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerL2L>::template Eval<Real>>(ker_l2l.Name().c_str(), DIM, std::pair<int,int>(ker_l2l.SrcDim(), ker_l2l.TrgDim()));
    for (auto& it : s2t_map) {
      it.second.setup_ker = true;
      it.second.setup_tree = true;
    }
  }
  #endif
}
template <class Real, Integer DIM> template <class KerS2M, class KerS2L> void ParticleFMM<Real,DIM>::AddSrc(const std::string& name, const KerS2M& ker_s2m, const KerS2L& ker_s2l) {
  SCTL_ASSERT_MSG(src_map.find(name) == src_map.end(), "Source name already exists.");
  src_map[name] = SrcData();
  auto& data = src_map[name];

  data.ker_s2m = (Iterator<char>)aligned_new<KerS2M>(1);
  data.ker_s2l = (Iterator<char>)aligned_new<KerS2L>(1);

  (*(Iterator<KerS2M>)data.ker_s2m) = ker_s2m;
  (*(Iterator<KerS2L>)data.ker_s2l) = ker_s2l;

  data.dim_src = ker_s2m.SrcDim();
  data.dim_mul_ch = ker_s2m.TrgDim();
  data.dim_loc_ch = ker_s2l.TrgDim();
  data.dim_normal = ker_s2m.NormalDim();
  SCTL_ASSERT(ker_s2m.CoordDim() == DIM);
  SCTL_ASSERT(ker_s2l.CoordDim() == DIM);
  SCTL_ASSERT(ker_s2l.SrcDim() == data.dim_src);
  SCTL_ASSERT(ker_s2l.NormalDim() == data.dim_normal);

  data.ker_s2m_eval = KerS2M::template Eval<Real,false>;
  data.ker_s2l_eval = KerS2L::template Eval<Real,false>;

  data.delete_ker_s2m = DeleteKer<KerS2M>;
  data.delete_ker_s2l = DeleteKer<KerS2L>;

  #ifdef SCTL_HAVE_PVFMM
  if (DIM == 3) {
    if (data.dim_normal) {
      data.pvfmm_ker_s2m = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerS2M,true>::template Eval<Real>, PVFMMKernelFn<KerS2M>::template Eval<Real>>(ker_s2m.Name().c_str(), DIM, std::pair<int,int>(ker_s2m.SrcDim(), ker_s2m.TrgDim()));
      data.pvfmm_ker_s2l = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerS2L,true>::template Eval<Real>, PVFMMKernelFn<KerS2L>::template Eval<Real>>(ker_s2l.Name().c_str(), DIM, std::pair<int,int>(ker_s2l.SrcDim(), ker_s2l.TrgDim()));
    } else {
      data.pvfmm_ker_s2m = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerS2M>::template Eval<Real>>(ker_s2m.Name().c_str(), DIM, std::pair<int,int>(ker_s2m.SrcDim(), ker_s2m.TrgDim()));
      data.pvfmm_ker_s2l = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerS2L>::template Eval<Real>>(ker_s2l.Name().c_str(), DIM, std::pair<int,int>(ker_s2l.SrcDim(), ker_s2l.TrgDim()));
    }
    for (auto& it : s2t_map) {
      if (it.first.first != name) continue;
      it.second.setup_ker = true;
      it.second.setup_tree = true;
    }
  }
  #endif
}
template <class Real, Integer DIM> template <class KerM2T, class KerL2T> void ParticleFMM<Real,DIM>::AddTrg(const std::string& name, const KerM2T& ker_m2t, const KerL2T& ker_l2t) {
  SCTL_ASSERT_MSG(trg_map.find(name) == trg_map.end(), "Target name already exists.");
  trg_map[name] = TrgData();
  auto& data = trg_map[name];

  data.ker_m2t = (Iterator<char>)aligned_new<KerM2T>(1);
  data.ker_l2t = (Iterator<char>)aligned_new<KerL2T>(1);

  (*(Iterator<KerM2T>)data.ker_m2t) = ker_m2t;
  (*(Iterator<KerL2T>)data.ker_l2t) = ker_l2t;

  data.dim_trg = ker_l2t.TrgDim();
  data.dim_mul_eq = ker_m2t.SrcDim();
  data.dim_loc_eq = ker_l2t.SrcDim();
  SCTL_ASSERT(ker_m2t.CoordDim() == DIM);
  SCTL_ASSERT(ker_l2t.CoordDim() == DIM);
  SCTL_ASSERT(ker_m2t.TrgDim() == data.dim_trg);

  data.ker_m2t_eval = KerM2T::template Eval<Real,false>;
  data.ker_l2t_eval = KerL2T::template Eval<Real,false>;

  data.delete_ker_m2t = DeleteKer<KerM2T>;
  data.delete_ker_l2t = DeleteKer<KerL2T>;

  #ifdef SCTL_HAVE_PVFMM
  if (DIM == 3) {
    data.pvfmm_ker_m2t = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerM2T>::template Eval<Real>>(ker_m2t.Name().c_str(), DIM, std::pair<int,int>(ker_m2t.SrcDim(), ker_m2t.TrgDim()));
    data.pvfmm_ker_l2t = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerL2T>::template Eval<Real>>(ker_l2t.Name().c_str(), DIM, std::pair<int,int>(ker_l2t.SrcDim(), ker_l2t.TrgDim()));
    for (auto& it : s2t_map) {
      if (it.first.second != name) continue;
      it.second.setup_ker = true;
      it.second.setup_tree = true;
    }
  }
  #endif
}
template <class Real, Integer DIM> template <class KerS2T> void ParticleFMM<Real,DIM>::SetKernelS2T(const std::string& src_name, const std::string& trg_name, const KerS2T& ker_s2t) {
  SCTL_ASSERT_MSG(src_map.find(src_name) != src_map.end(), "Source name does not exists.");
  SCTL_ASSERT_MSG(trg_map.find(trg_name) != trg_map.end(), "Target name does not exists.");

  const auto name = std::make_pair(src_name, trg_name);
  if (s2t_map.find(name) != s2t_map.end()) DeleteS2T(src_name, trg_name);
  s2t_map[name] = S2TData();
  auto& data = s2t_map[name];

  data.ker_s2t = (Iterator<char>)aligned_new<KerS2T>(1);
  (*(Iterator<KerS2T>)data.ker_s2t) = ker_s2t;

  data.dim_src = ker_s2t.SrcDim();
  data.dim_trg = ker_s2t.TrgDim();
  data.dim_normal = ker_s2t.NormalDim();
  SCTL_ASSERT(ker_s2t.CoordDim() == DIM);

  data.ker_s2t_eval = KerS2T::template Eval<Real,false>;
  data.ker_s2t_eval_omp = KerS2T::template Eval<Real,true>;
  data.delete_ker_s2t = DeleteKer<KerS2T>;

  #ifdef SCTL_HAVE_PVFMM
  if (DIM == 3) {
    BuildSrcTrgScal(data, !comm_.Rank());
    if (data.dim_normal) {
      data.pvfmm_ker_s2t = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerS2T,true>::template Eval<Real>, PVFMMKernelFn<KerS2T>::template Eval<Real>>(ker_s2t.Name().c_str(), DIM, std::pair<int,int>(ker_s2t.SrcDim(), ker_s2t.TrgDim()));
    } else {
      data.pvfmm_ker_s2t = pvfmm::BuildKernel<Real, PVFMMKernelFn<KerS2T>::template Eval<Real>>(ker_s2t.Name().c_str(), DIM, std::pair<int,int>(ker_s2t.SrcDim(), ker_s2t.TrgDim()));
    }
  }
  data.tree_ptr = nullptr;
  data.setup_ker = true;
  data.setup_tree = true;
  #endif
}

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::DeleteSrc(const std::string& name) {
  SCTL_ASSERT_MSG(src_map.find(name) != src_map.end(), "Source name does not exist.");
  auto& data = src_map[name];

  data.delete_ker_s2m(data.ker_s2m);
  data.delete_ker_s2l(data.ker_s2l);
  src_map.erase(name);

  Vector<std::pair<std::string,std::string>> s2t_lst;
  for (auto& it : s2t_map) if (it.first.first == name) s2t_lst.PushBack(it.first);
  for (const auto& name : s2t_lst) DeleteS2T(name.first, name.second);
}
template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::DeleteTrg(const std::string& name) {
  SCTL_ASSERT_MSG(trg_map.find(name) != trg_map.end(), "Target name does not exist.");
  auto& data = trg_map[name];

  data.delete_ker_m2t(data.ker_m2t);
  data.delete_ker_l2t(data.ker_l2t);
  trg_map.erase(name);

  Vector<std::pair<std::string,std::string>> s2t_lst;
  for (auto& it : s2t_map) if (it.first.second == name) s2t_lst.PushBack(it.first);
  for (const auto& name : s2t_lst) DeleteS2T(name.first, name.second);
}
template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::DeleteS2T(const std::string& src_name, const std::string& trg_name) {
  const auto name = std::make_pair(src_name, trg_name);
  SCTL_ASSERT_MSG(s2t_map.find(name) != s2t_map.end(), "S2T kernel does not exist.");
  auto& data = s2t_map[name];

  #ifdef SCTL_HAVE_PVFMM
  if (data.tree_ptr) delete data.tree_ptr;
  #endif

  data.delete_ker_s2t(data.ker_s2t);
  s2t_map.erase(name);
}

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::SetSrcCoord(const std::string& name, const Vector<Real>& src_coord, const Vector<Real>& src_normal) {
  SCTL_ASSERT_MSG(src_map.find(name) != src_map.end(), "Target name does not exist.");
  auto& data = src_map[name];
  data.X = src_coord;
  data.Xn = src_normal;

  #ifdef SCTL_HAVE_PVFMM
  if (DIM == 3) {
    BoundingBox<DIM>(data.bbox, src_coord, comm_);
    for (auto& it : s2t_map) {
      if (it.first.first != name) continue;
      it.second.setup_tree = true;
    }
  }
  #endif
}
template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::SetSrcDensity(const std::string& name, const Vector<Real>& src_density) {
  SCTL_ASSERT_MSG(src_map.find(name) != src_map.end(), "Target name does not exist.");
  auto& data = src_map[name];
  data.F = src_density;
}
template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::SetTrgCoord(const std::string& name, const Vector<Real>& trg_coord) {
  SCTL_ASSERT_MSG(trg_map.find(name) != trg_map.end(), "Target name does not exist.");
  auto& data = trg_map[name];
  data.X = trg_coord;

  #ifdef SCTL_HAVE_PVFMM
  if (DIM == 3) {
    BoundingBox<DIM>(data.bbox, trg_coord, comm_);
    for (auto& it : s2t_map) {
      if (it.first.second != name) continue;
      it.second.setup_tree = true;
    }
  }
  #endif
}

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::Eval(Vector<Real>& U, const std::string& trg_name) const {
  CheckKernelDims();

  #ifdef SCTL_HAVE_PVFMM
  EvalPVFMM(U, trg_name);
  #else
  EvalDirect(U, trg_name);
  #endif
}
template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::EvalDirect(Vector<Real>& U, const std::string& trg_name) const {
  const Integer rank = comm_.Rank();
  const Integer np = comm_.Size();

  SCTL_ASSERT_MSG(trg_map.find(trg_name) != trg_map.end(), "Target name does not exist.");
  const auto& trg_data = trg_map.at(trg_name);
  const Integer TrgDim = trg_data.dim_trg;
  const auto& Xt = trg_data.X;

  const Long Nt = Xt.Dim() / DIM;
  SCTL_ASSERT(Xt.Dim() == Nt * DIM);
  if (U.Dim() != Nt * TrgDim) {
    U.ReInit(Nt * TrgDim);
    U.SetZero();
  }

  for (auto& it : s2t_map) {
    if (it.first.second != trg_name) continue;
    const std::string src_name = it.first.first;

    SCTL_ASSERT_MSG(src_map.find(src_name) != src_map.end(), "Source name does not exist.");
    const auto& src_data = src_map.at(src_name);
    const Integer SrcDim = src_data.dim_src;
    const Integer NorDim = src_data.dim_normal;
    const auto& Xs = src_data.X;
    const auto& F = src_data.F;

    const Vector<Real> Xn_dummy;
    const auto& Xn = (NorDim ? src_data.Xn : Xn_dummy);

    const Long Ns = Xs.Dim() / DIM;
    SCTL_ASSERT(Xs.Dim() == Ns * DIM);
    SCTL_ASSERT(F.Dim() == Ns * SrcDim);
    SCTL_ASSERT(Xn.Dim() == Ns * NorDim);

    Vector<Real> Xs_, Xn_, F_;
    for (Long i = 0; i < np; i++) {
      auto send_recv_vec = [this,rank,np](Vector<Real>& X_, const Vector<Real>& X, Integer offset){
        Integer send_partner = (rank + offset) % np;
        Integer recv_partner = (rank + np - offset) % np;

        Long send_cnt = X.Dim(), recv_cnt = 0;
        void* recv_req = comm_.Irecv(     Ptr2Itr<Long>(&recv_cnt,1), 1, recv_partner, offset);
        void* send_req = comm_.Isend(Ptr2ConstItr<Long>(&send_cnt,1), 1, send_partner, offset);
        comm_.Wait(recv_req);
        comm_.Wait(send_req);

        X_.ReInit(recv_cnt);
        recv_req = comm_.Irecv(X_.begin(), recv_cnt, recv_partner, offset);
        send_req = comm_.Isend(X .begin(), send_cnt, send_partner, offset);
        comm_.Wait(recv_req);
        comm_.Wait(send_req);
      };
      send_recv_vec(Xs_, Xs, i);
      send_recv_vec(Xn_, Xn, i);
      send_recv_vec(F_ , F , i);
      it.second.ker_s2t_eval_omp(U, Xt, Xs_, Xn_, F_, digits_, it.second.ker_s2t);
    }
  }
}

template <class Real, Integer DIM> template <class Ker> void ParticleFMM<Real,DIM>::DeleteKer(Iterator<char> ker) {
  aligned_delete((Iterator<Ker>)ker);
}

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::CheckKernelDims() const {
  SCTL_ASSERT(fmm_ker.ker_m2m != NullIterator<char>());
  SCTL_ASSERT(fmm_ker.ker_m2l != NullIterator<char>());
  SCTL_ASSERT(fmm_ker.ker_l2l != NullIterator<char>());
  const Integer DimMulEq = fmm_ker.dim_mul_eq;
  const Integer DimMulCh = fmm_ker.dim_mul_ch;
  const Integer DimLocEq = fmm_ker.dim_loc_eq;
  const Integer DimLocCh = fmm_ker.dim_loc_ch;

  for (auto& src_it : src_map) {
    for (auto& trg_it : trg_map) {
      const auto name = std::make_pair(src_it.first, trg_it.first);
      SCTL_ASSERT_MSG(s2t_map.find(name) != s2t_map.end(), "S2T kernel for " + name.first + "-" + name.second + " was not provided.");
    }
  }

  for (auto& it : s2t_map) {
    const auto& src_name = it.first.first;
    const auto& trg_name = it.first.second;
    const Integer SrcDim = it.second.dim_src;
    const Integer TrgDim = it.second.dim_trg;
    const Integer NormalDim = it.second.dim_normal;

    SCTL_ASSERT_MSG(src_map.find(src_name) != src_map.end(), "Source name does not exist.");
    SCTL_ASSERT_MSG(trg_map.find(trg_name) != trg_map.end(), "Source name does not exist.");
    const auto& src_data = src_map.at(src_name);
    const auto& trg_data = trg_map.at(trg_name);

    SCTL_ASSERT(trg_data.dim_trg == TrgDim);
    SCTL_ASSERT(src_data.dim_src == SrcDim);
    SCTL_ASSERT(src_data.dim_normal == NormalDim);

    SCTL_ASSERT(src_data.dim_mul_ch == DimMulCh);
    SCTL_ASSERT(src_data.dim_loc_ch == DimLocCh);
    SCTL_ASSERT(trg_data.dim_mul_eq == DimMulEq);
    SCTL_ASSERT(trg_data.dim_loc_eq == DimLocEq);
  }
}

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::BuildSrcTrgScal(const S2TData& data, bool verbose) {
  const StaticArray<Integer,2> kdim{data.dim_src, data.dim_trg};
  const Integer dim_normal = data.dim_normal;
  const Real eps=machine_eps<Real>();

  auto BuildMatrix = [&data,&kdim,&dim_normal](Matrix<Real>& M, const Vector<Real>& src_X, const Vector<Real>& src_Xn, const Vector<Real>& trg_X) {
    const Long N = trg_X.Dim() / DIM;
    SCTL_ASSERT(trg_X.Dim() == N * DIM);
    SCTL_ASSERT(src_X.Dim() == DIM);
    SCTL_ASSERT(src_Xn.Dim() == dim_normal);

    M.ReInit(N, kdim[0]*kdim[1]);
    Vector<Real> F(kdim[0]), U(N*kdim[1]); F = 0;
    for (Integer k0 = 0; k0 < kdim[0]; k0++) {
      U = 0;
      F[k0] = 1;
      data.ker_s2t_eval(U, trg_X, src_X, src_Xn, F, -1, data.ker_s2t);
      for (Long i = 0; i < N; i++) {
        for (Integer k1 = 0; k1 < kdim[1]; k1++) {
          M[i][k0*kdim[1]+k1] = U[i*kdim[1]+k1];
        }
      }
      F[k0] = 0;
    }
  };

  Vector<Real>& src_scal_exp = data.src_scal_exp;
  Vector<Real>& trg_scal_exp = data.trg_scal_exp;
  src_scal_exp.ReInit(kdim[0]); src_scal_exp.SetZero();
  trg_scal_exp.ReInit(kdim[1]); trg_scal_exp.SetZero();

  Real scal=1.0;
  bool scale_invar = true;
  if (kdim[0]*kdim[1] > 0) { // Determine scaling
    Long N = 1024;
    Real eps_ = N * eps;

    Matrix<Real> M1(N, kdim[0]*kdim[1]);
    Vector<Real> src_coord(DIM), src_normal(dim_normal), trg_coord1(N*DIM);
    for (Integer i = 0; i < dim_normal; i++) src_normal[i] = (Real)(drand48() - 0.5);
    for (Integer i = 0; i < DIM; i++) src_coord[i] = 0;
    while (true) {
      Real inv_scal = 1/scal, abs_sum = 0;
      for (Long i = 0; i < N/2; i++) {
        for (Integer j = 0; j < DIM; j++) {
          trg_coord1[i*DIM+j] = (Real)(drand48()-0.5) * scal;
        }
      }
      for (Long i = N/2; i < N; i++) {
        for (Integer j = 0; j < DIM; j++) {
          trg_coord1[i*DIM+j] = (Real)(drand48()-0.5) * inv_scal;
        }
      }
      BuildMatrix(M1, src_coord, src_normal, trg_coord1);
      for (const auto& a : M1) abs_sum += fabs<Real>(a);
      if (abs_sum > sqrt<Real>(eps) || scal < eps) break;
      scal = scal * (Real)0.5;
    }

    Vector<Real> trg_coord2 = trg_coord1*(Real)0.5;
    Matrix<Real> M2(N,kdim[0]*kdim[1]);
    BuildMatrix(M2, src_coord, src_normal, trg_coord2);

    Real max_val = 0;
    Matrix<Real> M_scal(kdim[0], kdim[1]);
    for (Integer i = 0; i < kdim[0]*kdim[1]; i++) {
      Real dot11 = 0, dot22 = 0;
      for(Long j = 0; j < N; j++) {
        dot11 += M1[j][i] * M1[j][i];
        dot22 += M2[j][i] * M2[j][i];
      }
      max_val = std::max<Real>(max_val, dot11);
      max_val = std::max<Real>(max_val, dot22);
    }
    for (Integer i = 0; i < kdim[0]*kdim[1]; i++) {
      Real dot11 = 0, dot12 = 0, dot22 = 0;
      for (Long j = 0; j < N; j++) {
        dot11 += M1[j][i] * M1[j][i];
        dot12 += M1[j][i] * M2[j][i];
        dot22 += M2[j][i] * M2[j][i];
      }
      if (dot11>max_val*eps && dot22>max_val*eps) {
        Real s = dot12 / dot11;
        M_scal[0][i] = log<Real>(s) / log<Real>(2.0);
        Real err = sqrt<Real>((Real)0.5*(dot22/dot11)/(s*s) - (Real)0.5);
        if (err > eps_) {
          scale_invar = false;
          M_scal[0][i] = 0.0;
        }
        //assert(M_scal[0][i]>=0.0); // Kernel function must decay
      }else if(dot11>max_val*eps || dot22>max_val*eps) {
        scale_invar = false;
        M_scal[0][i] = 0.0;
      }else{
        M_scal[0][i] = -1;
      }
    }

    if (scale_invar) {
      Matrix<Real> b(kdim[0]*kdim[1]+1,1); b.SetZero();
      for (Integer i = 0; i < kdim[0]*kdim[1]; i++) b[i][0] = M_scal[0][i];

      Matrix<Real> M(kdim[0]*kdim[1]+1, kdim[0]+kdim[1]); M.SetZero();
      M[kdim[0]*kdim[1]][0] = 1;
      for (Integer i0 = 0; i0 < kdim[0]; i0++) {
        for (Integer i1 = 0; i1 < kdim[1]; i1++) {
          Integer j = i0*kdim[1] + i1;
          if (b[j][0] > 0) {
            M[j][ 0+        i0] = 1;
            M[j][i1+kdim[0]] = 1;
          }
        }
      }

      Matrix<Real> x = M.pinv() * b;
      for (Integer i = 0; i < kdim[0]; i++) { // Set src_scal_exp
        src_scal_exp[i] = x[i][0];
      }
      for (Integer i = 0; i < kdim[1]; i++) { // Set src_scal_exp
        trg_scal_exp[i] = x[kdim[0]+i][0];
      }
      for (Integer i0 = 0; i0 < kdim[0]; i0++) { // Verify
        for (Integer i1 = 0; i1 < kdim[1]; i1++) {
          if (M_scal[i0][i1] >= 0) {
            if (fabs<Real>(src_scal_exp[i0] + trg_scal_exp[i1] - M_scal[i0][i1]) > eps_) {
              scale_invar = false;
            }
          }
        }
      }
    }
    if (!scale_invar) {
      src_scal_exp.SetZero();
      trg_scal_exp.SetZero();
    }
  }

  #ifdef SCTL_VERBOSE
  if (verbose && scale_invar && kdim[0]*kdim[1] > 0) {
    std::cout<<"Scaling Matrix :\n";
    Matrix<Real> Src(kdim[0],1);
    Matrix<Real> Trg(1,kdim[1]);
    for(Integer i=0;i<kdim[0];i++) Src[i][0]=pow<Real>(2.0,src_scal_exp[i]);
    for(Integer i=0;i<kdim[1];i++) Trg[0][i]=pow<Real>(2.0,trg_scal_exp[i]);
    std::cout<<Src*Trg<<'\n';
  }
  #endif
}

#ifdef SCTL_HAVE_PVFMM
template <class Real, Integer DIM> template <class SCTLKernel, bool use_dummy_normal> struct ParticleFMM<Real,DIM>::PVFMMKernelFn : public pvfmm::GenericKernel<PVFMMKernelFn<SCTLKernel, use_dummy_normal>> {
  static const int FLOPS = SCTLKernel::FLOPS() + 2*SCTLKernel::SrcDim()*SCTLKernel::TrgDim();;

  template <class ValueType> static ValueType ScaleFactor();

  template <class VecType, int digits> static void uKerEval(VecType (&u)[SCTLKernel::TrgDim()], const VecType (&r)[SCTLKernel::CoordDim()], const VecType (&f)[SCTLKernel::SrcDim()+(use_dummy_normal?0:SCTLKernel::NormalDim())], const void* ctx_ptr);
};

template <class Real, Integer DIM> template <class SCTLKernel, bool use_dummy_normal> template <class ValueType> ValueType ParticleFMM<Real,DIM>::PVFMMKernelFn<SCTLKernel,use_dummy_normal>::ScaleFactor() {
  return SCTLKernel::template uKerScaleFactor<ValueType>();
}

template <class Real, Integer DIM> template <class SCTLKernel, bool use_dummy_normal> template <class VecType, int digits> void ParticleFMM<Real,DIM>::PVFMMKernelFn<SCTLKernel,use_dummy_normal>::uKerEval(VecType (&u)[SCTLKernel::TrgDim()], const VecType (&r)[SCTLKernel::CoordDim()], const VecType (&f)[SCTLKernel::SrcDim()+(use_dummy_normal?0:SCTLKernel::NormalDim())], const void* ctx_ptr) {
  constexpr Integer KDIM0 = SCTLKernel::SrcDim();
  constexpr Integer KDIM1 = SCTLKernel::TrgDim();
  constexpr Integer N_DIM = SCTLKernel::NormalDim();
  constexpr Integer N_DIM_ = (N_DIM?N_DIM:1);

  VecType Xn[N_DIM_], K[KDIM0][KDIM1];
  for (Integer i = 0; i < N_DIM; i++) { // Set Xn
    Xn[i] = (use_dummy_normal ? VecType((typename VecType::ScalarType)0) : f[KDIM0+i]);
  }
  SCTLKernel::template uKerMatrix<digits>(K, r, Xn, ctx_ptr);
  for (Integer k0 = 0; k0 < KDIM0; k0++) { // u <-- K * f
    for (Integer k1 = 0; k1 < KDIM1; k1++) {
      u[k1] = FMA(K[k0][k1], f[k0], u[k1]);
    }
  }
}

template <class Real, Integer DIM> void ParticleFMM<Real,DIM>::EvalPVFMM(Vector<Real>& U, const std::string& trg_name) const {
  if (DIM != 3) return EvalDirect(U, trg_name); // PVFMM only supports 3D

  SCTL_ASSERT_MSG(trg_map.find(trg_name) != trg_map.end(), "Target name does not exist.");
  const auto& trg_data = trg_map.at(trg_name);
  const Integer TrgDim = trg_data.dim_trg;
  const auto& Xt = trg_data.X;

  const Long Nt = Xt.Dim() / DIM;
  SCTL_ASSERT(Xt.Dim() == Nt * DIM);
  { // User EvalDirect for small problems
    StaticArray<Long,2> cnt{Nt,0};
    comm_.Allreduce<Long>(cnt+0, cnt+1, 1, Comm::CommOp::SUM);
    if (cnt[1] < 40000) return EvalDirect(U, trg_name);
  }
  if (U.Dim() != Nt * TrgDim) {
    U.ReInit(Nt * TrgDim);
    U.SetZero();
  }

  for (auto& it : s2t_map) {
    if (it.first.second != trg_name) continue;
    const std::string src_name = it.first.first;
    const auto& s2t_data = it.second;

    SCTL_ASSERT_MSG(src_map.find(src_name) != src_map.end(), "Source name does not exist.");
    const auto& src_data = src_map.at(src_name);
    const Integer SrcDim = src_data.dim_src;
    const Integer NorDim = src_data.dim_normal;
    const auto& Xs = src_data.X;
    const auto& F = src_data.F;

    const Vector<Real> Xn_dummy;
    const auto& Xn = (NorDim ? src_data.Xn : Xn_dummy);

    const Long Ns = Xs.Dim() / DIM;
    SCTL_ASSERT(Xs.Dim() == Ns * DIM);
    SCTL_ASSERT(F.Dim() == Ns * SrcDim);
    SCTL_ASSERT(Xn.Dim() == Ns * NorDim);
    if (!Ns) continue;

    { // Run FMM
      const Integer max_pts=3000, mult_order = ((Integer)(digits_*0.55))*2; // TODO: use better estimates

      pvfmm::PtFMM<Real>& fmm_ctx = s2t_data.fmm_ctx;
      if (s2t_data.setup_ker) { // Setup fmm_ctx
        pvfmm::Kernel<Real>& pvfmm_ker_s2t = s2t_data.pvfmm_ker_s2t;
        pvfmm_ker_s2t.k_m2m = &fmm_ker.pvfmm_ker_m2m;
        pvfmm_ker_s2t.k_m2l = &fmm_ker.pvfmm_ker_m2l;
        pvfmm_ker_s2t.k_l2l = &fmm_ker.pvfmm_ker_l2l;
        pvfmm_ker_s2t.k_m2t = &trg_data.pvfmm_ker_m2t;
        pvfmm_ker_s2t.k_l2t = &trg_data.pvfmm_ker_l2t;
        pvfmm_ker_s2t.k_s2m = &src_data.pvfmm_ker_s2m;
        pvfmm_ker_s2t.k_s2l = &src_data.pvfmm_ker_s2l;
        fmm_ctx.Initialize(mult_order, comm_.GetMPI_Comm(), &pvfmm_ker_s2t);
        s2t_data.setup_ker = false;
      }

      std::vector<Real> sl_den_, dl_den_;
      Vector<Real>& src_scal = s2t_data.src_scal;
      Vector<Real>& trg_scal = s2t_data.trg_scal;
      pvfmm::PtFMM_Tree<Real>*& tree_ptr = s2t_data.tree_ptr;
      if (s2t_data.setup_tree) { // Setup tree_ptr, src_scal, trg_scal
        auto& bbox_scale = s2t_data.bbox_scale;
        auto& bbox_offset = s2t_data.bbox_offset;
        { // Set bbox_scale, bbox_offset
          StaticArray<Real,DIM*2> bbox;
          for (Integer k = 0; k < DIM; k++) {
            bbox[k*2+0] = std::min<Real>(src_data.bbox[k*2+0], trg_data.bbox[k*2+0]);
            bbox[k*2+1] = std::max<Real>(src_data.bbox[k*2+1], trg_data.bbox[k*2+1]);
          }

          Real bbox_len = 0;
          for (Integer k = 0; k < DIM; k++) {
            bbox_offset[k] = (bbox[k*2+0] + bbox[k*2+1])/2;
            bbox_len = std::max<Real>(bbox_len, bbox[k*2+1]-bbox[k*2+0]);
          }

          bbox_scale = 1/(bbox_len*(Real)1.1); // extra 5% padding so that points are not on boundary
          for (Integer k = 0; k < DIM; k++) {
            bbox_offset[k] -= 1/(2*bbox_scale);
          }
        }
        { // Set src_scal, trg_scal
          src_scal.ReInit(SrcDim);
          trg_scal.ReInit(TrgDim);
          const Vector<Real>& src_scal_exp = s2t_data.src_scal_exp;
          const Vector<Real>& trg_scal_exp = s2t_data.trg_scal_exp;
          for (Integer i = 0; i < SrcDim; i++) src_scal[i] = pow<Real>(bbox_scale, src_scal_exp[i]);
          for (Integer i = 0; i < TrgDim; i++) trg_scal[i] = pow<Real>(bbox_scale, trg_scal_exp[i]);
        }

        std::vector<Real> sl_coord_, dl_coord_, trg_coord_(Nt*DIM);
        auto& src_coord = (NorDim ? dl_coord_ : sl_coord_);
        src_coord.resize(Ns * DIM);
        for (Long i = 0; i < Ns; i++) {
          for (Integer k = 0; k < DIM; k++) {
            src_coord[i*DIM+k] = (Xs[i*DIM+k] - bbox_offset[k]) * bbox_scale;
          }
        }
        for (Long i = 0; i < Nt; i++) {
          for (Integer k = 0; k < DIM; k++) {
            trg_coord_[i*DIM+k] = (Xt[i*DIM+k] - bbox_offset[k]) * bbox_scale;
          }
        }

        if (tree_ptr) delete tree_ptr;
        sl_den_.resize(NorDim ? 0 : Ns*SrcDim);
        dl_den_.resize(NorDim ? Ns*(SrcDim+NorDim) : 0);
        tree_ptr = PtFMM_CreateTree(sl_coord_, sl_den_, dl_coord_, dl_den_, trg_coord_, comm_.GetMPI_Comm(), max_pts, pvfmm::FreeSpace);
        tree_ptr->SetupFMM(&fmm_ctx);
        s2t_data.setup_tree = false;
      } else {
        tree_ptr->ClearFMMData();
      }
      if (NorDim) { // Set sl_den_, dl_den_
        dl_den_.resize(Ns*(SrcDim+NorDim));
        for (Long i = 0; i < Ns; i++) {
          for (Long j = 0; j < SrcDim; j++) {
            dl_den_[i*(SrcDim+NorDim) + j] = F[i*SrcDim+j] * src_scal[j];
          }
          for (Long j = 0; j < NorDim; j++) {
            dl_den_[i*(SrcDim+NorDim) + SrcDim+j] = Xn[i*NorDim+j];
          }
        }
      } else {
        sl_den_.resize(Ns*SrcDim);
        for (Long i = 0; i < Ns; i++) {
          for (Long j = 0; j < SrcDim; j++) {
            sl_den_[i*SrcDim+j] = F[i*SrcDim+j] * src_scal[j];
          }
        }
      }

      std::vector<Real> trg_value;
      PtFMM_Evaluate(tree_ptr, trg_value, Nt, &sl_den_, &dl_den_);
      //pvfmm::Profile::print(&comm_.GetMPI_Comm());
      SCTL_ASSERT(trg_value.size() == (size_t)(Nt*TrgDim));
      for (Long i = 0; i < Nt; i++) {
        for (Long j = 0; j < TrgDim; j++) {
          U[i*TrgDim+j] += trg_value[i*TrgDim+j] * trg_scal[j];
        }
      }
    }
  }
}
#endif

}  // end namespace

