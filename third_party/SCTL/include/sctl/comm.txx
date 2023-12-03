#include <type_traits>
#include SCTL_INCLUDE(ompUtils.hpp)
#include SCTL_INCLUDE(vector.hpp)

namespace SCTL_NAMESPACE {

inline Comm::Comm() {
#ifdef SCTL_HAVE_MPI
  Init(MPI_COMM_SELF);
#endif
}

inline Comm::Comm(const Comm& c) {
#ifdef SCTL_HAVE_MPI
  Init(c.mpi_comm_);
#endif
}

inline Comm Comm::Self() {
#ifdef SCTL_HAVE_MPI
  Comm comm_self(MPI_COMM_SELF);
  return comm_self;
#else
  Comm comm_self;
  return comm_self;
#endif
}

inline Comm Comm::World() {
#ifdef SCTL_HAVE_MPI
  Comm comm_world(MPI_COMM_WORLD);
  return comm_world;
#else
  Comm comm_self;
  return comm_self;
#endif
}

inline Comm& Comm::operator=(const Comm& c) {
#ifdef SCTL_HAVE_MPI
  #pragma omp critical(SCTL_COMM_DUP)
  MPI_Comm_free(&mpi_comm_);
  Init(c.mpi_comm_);
#endif
  return *this;
}

inline Comm::~Comm() {
#ifdef SCTL_HAVE_MPI
  while (!req.empty()) {
    delete (Vector<MPI_Request>*)req.top();
    req.pop();
  }
  #pragma omp critical(SCTL_COMM_DUP)
  MPI_Comm_free(&mpi_comm_);
#endif
}

inline Comm Comm::Split(Integer clr) const {
#ifdef SCTL_HAVE_MPI
  MPI_Comm new_comm;
  #pragma omp critical(SCTL_COMM_DUP)
  MPI_Comm_split(mpi_comm_, clr, mpi_rank_, &new_comm);
  Comm c(new_comm);
  #pragma omp critical(SCTL_COMM_DUP)
  MPI_Comm_free(&new_comm);
  return c;
#else
  Comm c;
  return c;
#endif
}

inline Integer Comm::Rank() const {
#ifdef SCTL_HAVE_MPI
  return mpi_rank_;
#else
  return 0;
#endif
}

inline Integer Comm::Size() const {
#ifdef SCTL_HAVE_MPI
  return mpi_size_;
#else
  return 1;
#endif
}

inline void Comm::Barrier() const {
#ifdef SCTL_HAVE_MPI
  MPI_Barrier(mpi_comm_);
#endif
}

template <class SType> void* Comm::Isend(ConstIterator<SType> sbuf, Long scount, Integer dest, Integer tag) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!scount) return nullptr;
  Vector<MPI_Request>& request = *NewReq();
  request.ReInit(1);

  SCTL_UNUSED(sbuf[0]         );
  SCTL_UNUSED(sbuf[scount - 1]);
#ifndef NDEBUG
  MPI_Issend(&sbuf[0], scount, CommDatatype<SType>::value(), dest, tag, mpi_comm_, &request[0]);
#else
  MPI_Isend(&sbuf[0], scount, CommDatatype<SType>::value(), dest, tag, mpi_comm_, &request[0]);
#endif
  return &request;
#else
  auto it = recv_req.find(tag);
  if (it == recv_req.end()) {
    send_req.insert(std::pair<Integer, ConstIterator<char>>(tag, (ConstIterator<char>)sbuf));
  } else {
    memcopy(it->second, (ConstIterator<char>)sbuf, scount * sizeof(SType));
    recv_req.erase(it);
  }
  return nullptr;
#endif
}

template <class RType> void* Comm::Irecv(Iterator<RType> rbuf, Long rcount, Integer source, Integer tag) const {
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!rcount) return nullptr;
  Vector<MPI_Request>& request = *NewReq();
  request.ReInit(1);

  SCTL_UNUSED(rbuf[0]         );
  SCTL_UNUSED(rbuf[rcount - 1]);
  MPI_Irecv(&rbuf[0], rcount, CommDatatype<RType>::value(), source, tag, mpi_comm_, &request[0]);
  return &request;
#else
  auto it = send_req.find(tag);
  if (it == send_req.end()) {
    recv_req.insert(std::pair<Integer, Iterator<char>>(tag, (Iterator<char>)rbuf));
  } else {
    memcopy((Iterator<char>)rbuf, it->second, rcount * sizeof(RType));
    send_req.erase(it);
  }
  return nullptr;
#endif
}

inline void Comm::Wait(void* req_ptr) const {
#ifdef SCTL_HAVE_MPI
  if (req_ptr == nullptr) return;
  Vector<MPI_Request>& request = *(Vector<MPI_Request>*)req_ptr;
  // std::vector<MPI_Status> status(request.Dim());
  if (request.Dim()) MPI_Waitall(request.Dim(), &request[0], MPI_STATUSES_IGNORE);  //&status[0]);
  DelReq(&request);
#endif
}

template <class Type> void Comm::Bcast(Iterator<Type> buf, Long count, Long root) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!count) return;
  SCTL_UNUSED(buf[0]        );
  SCTL_UNUSED(buf[count - 1]);
  MPI_Bcast(&buf[0], count, CommDatatype<Type>::value(), root, mpi_comm_);
#endif
}

template <class SType, class RType> void Comm::Allgather(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, Long rcount) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
  if (scount) {
    SCTL_UNUSED(sbuf[0]         );
    SCTL_UNUSED(sbuf[scount - 1]);
  }
  if (rcount) {
    SCTL_UNUSED(rbuf[0]                  );
    SCTL_UNUSED(rbuf[rcount * Size() - 1]);
  }
#ifdef SCTL_HAVE_MPI
  MPI_Allgather((scount ? &sbuf[0] : nullptr), scount, CommDatatype<SType>::value(), (rcount ? &rbuf[0] : nullptr), rcount, CommDatatype<RType>::value(), mpi_comm_);
#else
  memcopy((Iterator<char>)rbuf, (ConstIterator<char>)sbuf, scount * sizeof(SType));
#endif
}

template <class SType, class RType> void Comm::Allgatherv(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  Vector<int> rcounts_(mpi_size_), rdispls_(mpi_size_);
  Long rcount_sum = 0;
#pragma omp parallel for schedule(static) reduction(+ : rcount_sum)
  for (Integer i = 0; i < mpi_size_; i++) {
    rcounts_[i] = rcounts[i];
    rdispls_[i] = rdispls[i];
    rcount_sum += rcounts[i];
  }
  if (scount) {
    SCTL_UNUSED(sbuf[0]         );
    SCTL_UNUSED(sbuf[scount - 1]);
  }
  if (rcount_sum) {
    SCTL_UNUSED(rbuf[0]             );
    SCTL_UNUSED(rbuf[rcount_sum - 1]);
  }
  MPI_Allgatherv((scount ? &sbuf[0] : nullptr), scount, CommDatatype<SType>::value(), (rcount_sum ? &rbuf[0] : nullptr), &rcounts_.begin()[0], &rdispls_.begin()[0], CommDatatype<RType>::value(), mpi_comm_);
#else
  memcopy((Iterator<char>)(rbuf + rdispls[0]), (ConstIterator<char>)sbuf, scount * sizeof(SType));
#endif
}

template <class SType, class RType> void Comm::Alltoall(ConstIterator<SType> sbuf, Long scount, Iterator<RType> rbuf, Long rcount) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (scount) {
    SCTL_UNUSED(sbuf[0]                  );
    SCTL_UNUSED(sbuf[scount * Size() - 1]);
  }
  if (rcount) {
    SCTL_UNUSED(rbuf[0]                  );
    SCTL_UNUSED(rbuf[rcount * Size() - 1]);
  }
  MPI_Alltoall((scount ? &sbuf[0] : nullptr), scount, CommDatatype<SType>::value(), (rcount ? &rbuf[0] : nullptr), rcount, CommDatatype<RType>::value(), mpi_comm_);
#else
  memcopy((Iterator<char>)rbuf, (ConstIterator<char>)sbuf, scount * sizeof(SType));
#endif
}

template <class SType, class RType> void* Comm::Ialltoallv_sparse(ConstIterator<SType> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<RType> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls, Integer tag) const {
  static_assert(std::is_trivially_copyable<SType>::value, "Data is not trivially copyable!");
  static_assert(std::is_trivially_copyable<RType>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  Integer request_count = 0;
  for (Integer i = 0; i < mpi_size_; i++) {
    if (rcounts[i]) request_count++;
    if (scounts[i]) request_count++;
  }
  if (!request_count) return nullptr;
  Vector<MPI_Request>& request = *NewReq();
  request.ReInit(request_count);
  Integer request_iter = 0;

  for (Integer i = 0; i < mpi_size_; i++) {
    if (rcounts[i]) {
      SCTL_UNUSED(rbuf[rdispls[i]]);
      SCTL_UNUSED(rbuf[rdispls[i] + rcounts[i] - 1]);
      MPI_Irecv(&rbuf[rdispls[i]], rcounts[i], CommDatatype<RType>::value(), i, tag, mpi_comm_, &request[request_iter]);
      request_iter++;
    }
  }
  for (Integer i = 0; i < mpi_size_; i++) {
    if (scounts[i]) {
      SCTL_UNUSED(sbuf[sdispls[i]]);
      SCTL_UNUSED(sbuf[sdispls[i] + scounts[i] - 1]);
      MPI_Isend(&sbuf[sdispls[i]], scounts[i], CommDatatype<SType>::value(), i, tag, mpi_comm_, &request[request_iter]);
      request_iter++;
    }
  }
  return &request;
#else
  memcopy((Iterator<char>)(rbuf + rdispls[0]), (ConstIterator<char>)(sbuf + sdispls[0]), scounts[0] * sizeof(SType));
  return nullptr;
#endif
}

template <class Type> void Comm::Alltoallv(ConstIterator<Type> sbuf, ConstIterator<Long> scounts, ConstIterator<Long> sdispls, Iterator<Type> rbuf, ConstIterator<Long> rcounts, ConstIterator<Long> rdispls) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  {  // Use Alltoallv_sparse of average connectivity<64
    Long connectivity = 0, glb_connectivity = 0;
#pragma omp parallel for schedule(static) reduction(+ : connectivity)
    for (Integer i = 0; i < mpi_size_; i++) {
      if (rcounts[i]) connectivity++;
    }
    Allreduce(Ptr2ConstItr<Long>(&connectivity, 1), Ptr2Itr<Long>(&glb_connectivity, 1), 1, CommOp::SUM);
    if (glb_connectivity < 64 * Size()) {
      void* mpi_req = Ialltoallv_sparse(sbuf, scounts, sdispls, rbuf, rcounts, rdispls, 0);
      Wait(mpi_req);
      return;
    }
  }

  {  // Use vendor MPI_Alltoallv
    //#ifndef ALLTOALLV_FIX
    Vector<int> scnt, sdsp, rcnt, rdsp;
    scnt.ReInit(mpi_size_);
    sdsp.ReInit(mpi_size_);
    rcnt.ReInit(mpi_size_);
    rdsp.ReInit(mpi_size_);
    Long stotal = 0, rtotal = 0;
#pragma omp parallel for schedule(static) reduction(+ : stotal, rtotal)
    for (Integer i = 0; i < mpi_size_; i++) {
      scnt[i] = scounts[i];
      sdsp[i] = sdispls[i];
      rcnt[i] = rcounts[i];
      rdsp[i] = rdispls[i];
      stotal += scounts[i];
      rtotal += rcounts[i];
    }

    MPI_Alltoallv((stotal ? &sbuf[0] : nullptr), &scnt[0], &sdsp[0], CommDatatype<Type>::value(), (rtotal ? &rbuf[0] : nullptr), &rcnt[0], &rdsp[0], CommDatatype<Type>::value(), mpi_comm_);
    return;
    //#endif
  }

// TODO: implement hypercube scheme
#else
  memcopy((Iterator<char>)(rbuf + rdispls[0]), (ConstIterator<char>)(sbuf + sdispls[0]), scounts[0] * sizeof(Type));
#endif
}

template <class Type> void Comm::Allreduce(ConstIterator<Type> sbuf, Iterator<Type> rbuf, Long count, CommOp op) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!count) return;
  MPI_Op mpi_op;
  switch (op) {
    case CommOp::SUM:
      mpi_op = CommDatatype<Type>::sum();
      break;
    case CommOp::MIN:
      mpi_op = CommDatatype<Type>::min();
      break;
    case CommOp::MAX:
      mpi_op = CommDatatype<Type>::max();
      break;
    default:
      mpi_op = MPI_OP_NULL;
      break;
  }
  SCTL_UNUSED(sbuf[0]        );
  SCTL_UNUSED(sbuf[count - 1]);
  SCTL_UNUSED(rbuf[0]        );
  SCTL_UNUSED(rbuf[count - 1]);
  MPI_Allreduce(&sbuf[0], &rbuf[0], count, CommDatatype<Type>::value(), mpi_op, mpi_comm_);
#else
  memcopy((Iterator<char>)rbuf, (ConstIterator<char>)sbuf, count * sizeof(Type));
#endif
}

template <class Type> void Comm::Scan(ConstIterator<Type> sbuf, Iterator<Type> rbuf, int count, CommOp op) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  if (!count) return;
  MPI_Op mpi_op;
  switch (op) {
    case CommOp::SUM:
      mpi_op = CommDatatype<Type>::sum();
      break;
    case CommOp::MIN:
      mpi_op = CommDatatype<Type>::min();
      break;
    case CommOp::MAX:
      mpi_op = CommDatatype<Type>::max();
      break;
    default:
      mpi_op = MPI_OP_NULL;
      break;
  }
  SCTL_UNUSED(sbuf[0]        );
  SCTL_UNUSED(sbuf[count - 1]);
  SCTL_UNUSED(rbuf[0]        );
  SCTL_UNUSED(rbuf[count - 1]);
  MPI_Scan(&sbuf[0], &rbuf[0], count, CommDatatype<Type>::value(), mpi_op, mpi_comm_);
#else
  memcopy((Iterator<char>)rbuf, (ConstIterator<char>)sbuf, count * sizeof(Type));
#endif
}

template <class Type> void Comm::PartitionW(Vector<Type>& nodeList, const Vector<Long>* wts_) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  Integer npes = Size();
  if (npes == 1) return;
  Long nlSize = nodeList.Dim();

  Vector<Long> wts;
  Long localWt = 0;
  if (wts_ == nullptr) {  // Construct arrays of wts.
    wts.ReInit(nlSize);
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < nlSize; i++) {
      wts[i] = 1;
    }
    localWt = nlSize;
  } else {
    wts.ReInit(nlSize, (Iterator<Long>)wts_->begin(), false);
#pragma omp parallel for reduction(+ : localWt)
    for (Long i = 0; i < nlSize; i++) {
      localWt += wts[i];
    }
  }

  Long off1 = 0, off2 = 0, totalWt = 0;
  {  // compute the total weight of the problem ...
    Allreduce<Long>(Ptr2ConstItr<Long>(&localWt, 1), Ptr2Itr<Long>(&totalWt, 1), 1, CommOp::SUM);
    Scan<Long>(Ptr2ConstItr<Long>(&localWt, 1), Ptr2Itr<Long>(&off2, 1), 1, CommOp::SUM);
    off1 = off2 - localWt;
  }

  Vector<Long> lscn;
  if (nlSize) {  // perform a local scan on the weights first ...
    lscn.ReInit(nlSize);
    lscn[0] = off1;
    omp_par::scan(wts.begin(), lscn.begin(), nlSize);
  }

  Vector<Long> sendSz, recvSz, sendOff, recvOff;
  sendSz.ReInit(npes);
  recvSz.ReInit(npes);
  sendOff.ReInit(npes);
  recvOff.ReInit(npes);
  sendSz.SetZero();

  if (nlSize > 0 && totalWt > 0) {  // Compute sendSz
    Long pid1 = (off1 * npes) / totalWt;
    Long pid2 = ((off2 + 1) * npes) / totalWt + 1;
    assert((totalWt * pid2) / npes >= off2);
    pid1 = (pid1 < 0 ? 0 : pid1);
    pid2 = (pid2 > npes ? npes : pid2);
#pragma omp parallel for schedule(static)
    for (Integer i = pid1; i < pid2; i++) {
      Long wt1 = (totalWt * (i)) / npes;
      Long wt2 = (totalWt * (i + 1)) / npes;
      Long start = std::lower_bound(lscn.begin(), lscn.begin() + nlSize, wt1, std::less<Long>()) - lscn.begin();
      Long end = std::lower_bound(lscn.begin(), lscn.begin() + nlSize, wt2, std::less<Long>()) - lscn.begin();
      if (i == 0) start = 0;
      if (i == npes - 1) end = nlSize;
      sendSz[i] = end - start;
    }
  } else {
    sendSz[0] = nlSize;
  }

  // Exchange sendSz, recvSz
  Alltoall<Long>(sendSz.begin(), 1, recvSz.begin(), 1);

  {  // Compute sendOff, recvOff
    sendOff[0] = 0;
    omp_par::scan(sendSz.begin(), sendOff.begin(), npes);
    recvOff[0] = 0;
    omp_par::scan(recvSz.begin(), recvOff.begin(), npes);
    assert(sendOff[npes - 1] + sendSz[npes - 1] == nlSize);
  }

  // perform All2All  ...
  Vector<Type> newNodes;
  newNodes.ReInit(recvSz[npes - 1] + recvOff[npes - 1]);
  void* mpi_req = Ialltoallv_sparse<Type>(nodeList.begin(), sendSz.begin(), sendOff.begin(), newNodes.begin(), recvSz.begin(), recvOff.begin());
  Wait(mpi_req);

  // reset the pointer ...
  nodeList.Swap(newNodes);
}

template <class Type> void Comm::PartitionN(Vector<Type>& v, Long N) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  Integer rank = Rank();
  Integer np = Size();
  if (np == 1) return;

  Vector<Long> v_cnt(np), v_dsp(np + 1);
  Vector<Long> N_cnt(np), N_dsp(np + 1);
  {  // Set v_cnt, v_dsp
    v_dsp[0] = 0;
    Long cnt = v.Dim();
    Allgather(Ptr2ConstItr<Long>(&cnt, 1), 1, v_cnt.begin(), 1);
    omp_par::scan(v_cnt.begin(), v_dsp.begin(), np);
    v_dsp[np] = v_cnt[np - 1] + v_dsp[np - 1];
  }
  {  // Set N_cnt, N_dsp
    N_dsp[0] = 0;
    Long cnt = N;
    Allgather(Ptr2ConstItr<Long>(&cnt, 1), 1, N_cnt.begin(), 1);
    omp_par::scan(N_cnt.begin(), N_dsp.begin(), np);
    N_dsp[np] = N_cnt[np - 1] + N_dsp[np - 1];
  }
  {  // Adjust for dof
    Long dof = (N_dsp[np] ? v_dsp[np] / N_dsp[np] : 0);
    assert(dof * N_dsp[np] == v_dsp[np]);
    if (dof == 0) return;

    if (dof != 1) {
#pragma omp parallel for schedule(static)
      for (Integer i = 0; i < np; i++) N_cnt[i] *= dof;
#pragma omp parallel for schedule(static)
      for (Integer i = 0; i <= np; i++) N_dsp[i] *= dof;
    }
  }

  Vector<Type> v_(N_cnt[rank]);
  {  // Set v_
    Vector<Long> scnt(np), sdsp(np);
    Vector<Long> rcnt(np), rdsp(np);
#pragma omp parallel for schedule(static)
    for (Integer i = 0; i < np; i++) {
      {  // Set scnt
        Long n0 = N_dsp[i + 0];
        Long n1 = N_dsp[i + 1];
        if (n0 < v_dsp[rank + 0]) n0 = v_dsp[rank + 0];
        if (n1 < v_dsp[rank + 0]) n1 = v_dsp[rank + 0];
        if (n0 > v_dsp[rank + 1]) n0 = v_dsp[rank + 1];
        if (n1 > v_dsp[rank + 1]) n1 = v_dsp[rank + 1];
        scnt[i] = n1 - n0;
      }
      {  // Set rcnt
        Long n0 = v_dsp[i + 0];
        Long n1 = v_dsp[i + 1];
        if (n0 < N_dsp[rank + 0]) n0 = N_dsp[rank + 0];
        if (n1 < N_dsp[rank + 0]) n1 = N_dsp[rank + 0];
        if (n0 > N_dsp[rank + 1]) n0 = N_dsp[rank + 1];
        if (n1 > N_dsp[rank + 1]) n1 = N_dsp[rank + 1];
        rcnt[i] = n1 - n0;
      }
    }
    sdsp[0] = 0;
    omp_par::scan(scnt.begin(), sdsp.begin(), np);
    rdsp[0] = 0;
    omp_par::scan(rcnt.begin(), rdsp.begin(), np);

    void* mpi_request = Ialltoallv_sparse(v.begin(), scnt.begin(), sdsp.begin(), v_.begin(), rcnt.begin(), rdsp.begin());
    Wait(mpi_request);
  }
  v.Swap(v_);
}

template <class Type, class Compare> void Comm::PartitionS(Vector<Type>& nodeList, const Type& splitter, Compare comp) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  Integer npes = Size();
  if (npes == 1) return;

  Vector<Type> mins(npes);
  Allgather(Ptr2ConstItr<Type>(&splitter, 1), 1, mins.begin(), 1);

  Vector<Long> scnt(npes), sdsp(npes);
  Vector<Long> rcnt(npes), rdsp(npes);
  {  // Compute scnt, sdsp
#pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      sdsp[i] = std::lower_bound(nodeList.begin(), nodeList.begin() + nodeList.Dim(), mins[i], comp) - nodeList.begin();
    }
#pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes - 1; i++) {
      scnt[i] = sdsp[i + 1] - sdsp[i];
    }
    scnt[npes - 1] = nodeList.Dim() - sdsp[npes - 1];
  }
  {  // Compute rcnt, rdsp
    rdsp[0] = 0;
    Alltoall(scnt.begin(), 1, rcnt.begin(), 1);
    omp_par::scan(rcnt.begin(), rdsp.begin(), npes);
  }
  {  // Redistribute nodeList
    Vector<Type> nodeList_(rdsp[npes - 1] + rcnt[npes - 1]);
    void* mpi_request = Ialltoallv_sparse(nodeList.begin(), scnt.begin(), sdsp.begin(), nodeList_.begin(), rcnt.begin(), rdsp.begin());
    Wait(mpi_request);
    nodeList.Swap(nodeList_);
  }
}

template <class Type> void Comm::SortScatterIndex(const Vector<Type>& key, Vector<Long>& scatter_index, const Type* split_key_) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  typedef SortPair<Type, Long> Pair_t;
  Integer npes = Size();

  Vector<Pair_t> parray(key.Dim());
  {  // Build global index.
    Long glb_dsp = 0;
    Long loc_size = key.Dim();
    Scan(Ptr2ConstItr<Long>(&loc_size, 1), Ptr2Itr<Long>(&glb_dsp, 1), 1, CommOp::SUM);
    glb_dsp -= loc_size;
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < loc_size; i++) {
      parray[i].key = key[i];
      parray[i].data = glb_dsp + i;
    }
  }

  Vector<Pair_t> psorted;
  HyperQuickSort(parray, psorted);

  if (npes > 1 && split_key_ != nullptr) {  // Partition data
    Vector<Type> split_key(npes);
    Allgather(Ptr2ConstItr<Type>(split_key_, 1), 1, split_key.begin(), 1);

    Vector<Long> sendSz(npes);
    Vector<Long> recvSz(npes);
    Vector<Long> sendOff(npes);
    Vector<Long> recvOff(npes);
    Long nlSize = psorted.Dim();
    sendSz.SetZero();

    if (nlSize > 0) {  // Compute sendSz
      // Determine processor range.
      Long pid1 = std::lower_bound(split_key.begin(), split_key.begin() + npes, psorted[0].key) - split_key.begin() - 1;
      Long pid2 = std::upper_bound(split_key.begin(), split_key.begin() + npes, psorted[nlSize - 1].key) - split_key.begin() + 1;
      pid1 = (pid1 < 0 ? 0 : pid1);
      pid2 = (pid2 > npes ? npes : pid2);

#pragma omp parallel for schedule(static)
      for (Integer i = pid1; i < pid2; i++) {
        Pair_t p1;
        p1.key = split_key[i];
        Pair_t p2;
        p2.key = split_key[i + 1 < npes ? i + 1 : i];
        Long start = std::lower_bound(psorted.begin(), psorted.begin() + nlSize, p1, std::less<Pair_t>()) - psorted.begin();
        Long end = std::lower_bound(psorted.begin(), psorted.begin() + nlSize, p2, std::less<Pair_t>()) - psorted.begin();
        if (i == 0) start = 0;
        if (i == npes - 1) end = nlSize;
        sendSz[i] = end - start;
      }
    }

    // Exchange sendSz, recvSz
    Alltoall<Long>(sendSz.begin(), 1, recvSz.begin(), 1);

    // compute offsets ...
    {  // Compute sendOff, recvOff
      sendOff[0] = 0;
      omp_par::scan(sendSz.begin(), sendOff.begin(), npes);
      recvOff[0] = 0;
      omp_par::scan(recvSz.begin(), recvOff.begin(), npes);
      assert(sendOff[npes - 1] + sendSz[npes - 1] == nlSize);
    }

    // perform All2All  ...
    Vector<Pair_t> newNodes(recvSz[npes - 1] + recvOff[npes - 1]);
    void* mpi_req = Ialltoallv_sparse<Pair_t>(psorted.begin(), sendSz.begin(), sendOff.begin(), newNodes.begin(), recvSz.begin(), recvOff.begin());
    Wait(mpi_req);

    // reset the pointer ...
    psorted.Swap(newNodes);
  }

  scatter_index.ReInit(psorted.Dim());
#pragma omp parallel for schedule(static)
  for (Long i = 0; i < psorted.Dim(); i++) {
    scatter_index[i] = psorted[i].data;
  }
}

template <class Type> void Comm::ScatterForward(Vector<Type>& data_, const Vector<Long>& scatter_index) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  typedef SortPair<Long, Long> Pair_t;
  Integer npes = Size(), rank = Rank();

  Long data_dim = 0;
  Long send_size = 0;
  Long recv_size = 0;
  {  // Set data_dim, send_size, recv_size
    recv_size = scatter_index.Dim();
    StaticArray<Long, 2> glb_size;
    StaticArray<Long, 2> loc_size;
    loc_size[0] = data_.Dim();
    loc_size[1] = recv_size;
    Allreduce<Long>(loc_size, glb_size, 2, CommOp::SUM);
    if (glb_size[0] == 0 || glb_size[1] == 0) return;  // Nothing to be done.
    data_dim = glb_size[0] / glb_size[1];
    SCTL_ASSERT(glb_size[0] == data_dim * glb_size[1]);
    send_size = data_.Dim() / data_dim;
  }

  if (npes == 1) {  // Scatter directly
    Vector<Type> data;
    data.ReInit(recv_size * data_dim);
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      Long src_indx = scatter_index[i] * data_dim;
      Long trg_indx = i * data_dim;
      for (Long j = 0; j < data_dim; j++) data[trg_indx + j] = data_[src_indx + j];
    }
    data_.Swap(data);
    return;
  }

  Vector<Long> glb_scan;
  {  // Global scan of data size.
    glb_scan.ReInit(npes);
    Long glb_rank = 0;
    Scan(Ptr2ConstItr<Long>(&send_size, 1), Ptr2Itr<Long>(&glb_rank, 1), 1, CommOp::SUM);
    glb_rank -= send_size;
    Allgather(Ptr2ConstItr<Long>(&glb_rank, 1), 1, glb_scan.begin(), 1);
  }

  Vector<Pair_t> psorted;
  {  // Sort scatter_index.
    psorted.ReInit(recv_size);
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      psorted[i].key = scatter_index[i];
      psorted[i].data = i;
    }
    omp_par::merge_sort(psorted.begin(), psorted.begin() + recv_size);
  }

  Vector<Long> recv_indx(recv_size);
  Vector<Long> send_indx(send_size);
  Vector<Long> sendSz(npes);
  Vector<Long> sendOff(npes);
  Vector<Long> recvSz(npes);
  Vector<Long> recvOff(npes);
  {  // Exchange send, recv indices.
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      recv_indx[i] = psorted[i].key;
    }

#pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      Long start = std::lower_bound(recv_indx.begin(), recv_indx.begin() + recv_size, glb_scan[i]) - recv_indx.begin();
      Long end = (i + 1 < npes ? std::lower_bound(recv_indx.begin(), recv_indx.begin() + recv_size, glb_scan[i + 1]) - recv_indx.begin() : recv_size);
      recvSz[i] = end - start;
      recvOff[i] = start;
    }

    Alltoall(recvSz.begin(), 1, sendSz.begin(), 1);
    sendOff[0] = 0;
    omp_par::scan(sendSz.begin(), sendOff.begin(), npes);
    assert(sendOff[npes - 1] + sendSz[npes - 1] == send_size);

    Alltoallv(recv_indx.begin(), recvSz.begin(), recvOff.begin(), send_indx.begin(), sendSz.begin(), sendOff.begin());
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      assert(send_indx[i] >= glb_scan[rank]);
      send_indx[i] -= glb_scan[rank];
      assert(send_indx[i] < send_size);
    }
  }

  Vector<Type> send_buff;
  {  // Prepare send buffer
    send_buff.ReInit(send_size * data_dim);
    ConstIterator<Type> data = data_.begin();
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      Long src_indx = send_indx[i] * data_dim;
      Long trg_indx = i * data_dim;
      for (Long j = 0; j < data_dim; j++) send_buff[trg_indx + j] = data[src_indx + j];
    }
  }

  Vector<Type> recv_buff;
  {  // All2Allv
    recv_buff.ReInit(recv_size * data_dim);
#pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      sendSz[i] *= data_dim;
      sendOff[i] *= data_dim;
      recvSz[i] *= data_dim;
      recvOff[i] *= data_dim;
    }
    Alltoallv(send_buff.begin(), sendSz.begin(), sendOff.begin(), recv_buff.begin(), recvSz.begin(), recvOff.begin());
  }

  {  // Build output data.
    data_.ReInit(recv_size * data_dim);
    Iterator<Type> data = data_.begin();
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      Long src_indx = i * data_dim;
      Long trg_indx = psorted[i].data * data_dim;
      for (Long j = 0; j < data_dim; j++) data[trg_indx + j] = recv_buff[src_indx + j];
    }
  }
}

template <class Type> void Comm::ScatterReverse(Vector<Type>& data_, const Vector<Long>& scatter_index_, Long loc_size_) const {
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
  typedef SortPair<Long, Long> Pair_t;
  Integer npes = Size(), rank = Rank();

  Long data_dim = 0;
  Long send_size = 0;
  Long recv_size = 0;
  {  // Set data_dim, send_size, recv_size
    recv_size = loc_size_;
    StaticArray<Long, 3> glb_size;
    StaticArray<Long, 3> loc_size;
    loc_size[0] = data_.Dim();
    loc_size[1] = scatter_index_.Dim();
    loc_size[2] = recv_size;
    Allreduce<Long>(loc_size, glb_size, 3, CommOp::SUM);
    if (glb_size[0] == 0 || glb_size[1] == 0) return;  // Nothing to be done.

    SCTL_ASSERT(glb_size[0] % glb_size[1] == 0);
    data_dim = glb_size[0] / glb_size[1];

    SCTL_ASSERT(loc_size[0] % data_dim == 0);
    send_size = loc_size[0] / data_dim;

    if (glb_size[0] != glb_size[2] * data_dim) {
      recv_size = (((rank + 1) * (glb_size[0] / data_dim)) / npes) - ((rank * (glb_size[0] / data_dim)) / npes);
    }
  }

  if (npes == 1) {  // Scatter directly
    Vector<Type> data;
    data.ReInit(recv_size * data_dim);
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      Long src_indx = i * data_dim;
      Long trg_indx = scatter_index_[i] * data_dim;
      for (Long j = 0; j < data_dim; j++) data[trg_indx + j] = data_[src_indx + j];
    }
    data_.Swap(data);
    return;
  }

  Vector<Long> scatter_index;
  {
    StaticArray<Long, 2> glb_rank;
    StaticArray<Long, 3> glb_size;
    StaticArray<Long, 2> loc_size;
    loc_size[0] = data_.Dim() / data_dim;
    loc_size[1] = scatter_index_.Dim();
    Scan<Long>(loc_size, glb_rank, 2, CommOp::SUM);
    Allreduce<Long>(loc_size, glb_size, 2, CommOp::SUM);
    SCTL_ASSERT(glb_size[0] == glb_size[1]);
    glb_rank[0] -= loc_size[0];
    glb_rank[1] -= loc_size[1];

    Vector<Long> glb_scan0(npes + 1);
    Vector<Long> glb_scan1(npes + 1);
    Allgather<Long>(glb_rank + 0, 1, glb_scan0.begin(), 1);
    Allgather<Long>(glb_rank + 1, 1, glb_scan1.begin(), 1);
    glb_scan0[npes] = glb_size[0];
    glb_scan1[npes] = glb_size[1];

    if (loc_size[0] != loc_size[1] || glb_rank[0] != glb_rank[1]) {  // Repartition scatter_index
      scatter_index.ReInit(loc_size[0]);

      Vector<Long> send_dsp(npes + 1);
      Vector<Long> recv_dsp(npes + 1);
#pragma omp parallel for schedule(static)
      for (Integer i = 0; i <= npes; i++) {
        send_dsp[i] = std::min(std::max(glb_scan0[i], glb_rank[1]), glb_rank[1] + loc_size[1]) - glb_rank[1];
        recv_dsp[i] = std::min(std::max(glb_scan1[i], glb_rank[0]), glb_rank[0] + loc_size[0]) - glb_rank[0];
      }

      // Long commCnt=0;
      Vector<Long> send_cnt(npes + 0);
      Vector<Long> recv_cnt(npes + 0);
#pragma omp parallel for schedule(static)  // reduction(+:commCnt)
      for (Integer i = 0; i < npes; i++) {
        send_cnt[i] = send_dsp[i + 1] - send_dsp[i];
        recv_cnt[i] = recv_dsp[i + 1] - recv_dsp[i];
        // if(send_cnt[i] && i!=rank) commCnt++;
        // if(recv_cnt[i] && i!=rank) commCnt++;
      }

      void* mpi_req = Ialltoallv_sparse<Long>(scatter_index_.begin(), send_cnt.begin(), send_dsp.begin(), scatter_index.begin(), recv_cnt.begin(), recv_dsp.begin(), 0);
      Wait(mpi_req);
    } else {
      scatter_index.ReInit(scatter_index_.Dim(), (Iterator<Long>)scatter_index_.begin(), false);
    }
  }

  Vector<Long> glb_scan(npes);
  {  // Global data size.
    Long glb_rank = 0;
    Scan(Ptr2ConstItr<Long>(&recv_size, 1), Ptr2Itr<Long>(&glb_rank, 1), 1, CommOp::SUM);
    glb_rank -= recv_size;
    Allgather(Ptr2ConstItr<Long>(&glb_rank, 1), 1, glb_scan.begin(), 1);
  }

  Vector<Pair_t> psorted(send_size);
  {  // Sort scatter_index.
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      psorted[i].key = scatter_index[i];
      psorted[i].data = i;
    }
    omp_par::merge_sort(psorted.begin(), psorted.begin() + send_size);
  }

  Vector<Long> recv_indx(recv_size);
  Vector<Long> send_indx(send_size);
  Vector<Long> sendSz(npes);
  Vector<Long> sendOff(npes);
  Vector<Long> recvSz(npes);
  Vector<Long> recvOff(npes);
  {  // Exchange send, recv indices.
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      send_indx[i] = psorted[i].key;
    }

#pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      Long start = std::lower_bound(send_indx.begin(), send_indx.begin() + send_size, glb_scan[i]) - send_indx.begin();
      Long end = (i + 1 < npes ? std::lower_bound(send_indx.begin(), send_indx.begin() + send_size, glb_scan[i + 1]) - send_indx.begin() : send_size);
      sendSz[i] = end - start;
      sendOff[i] = start;
    }

    Alltoall(sendSz.begin(), 1, recvSz.begin(), 1);
    recvOff[0] = 0;
    omp_par::scan(recvSz.begin(), recvOff.begin(), npes);
    assert(recvOff[npes - 1] + recvSz[npes - 1] == recv_size);

    Alltoallv(send_indx.begin(), sendSz.begin(), sendOff.begin(), recv_indx.begin(), recvSz.begin(), recvOff.begin());
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      assert(recv_indx[i] >= glb_scan[rank]);
      recv_indx[i] -= glb_scan[rank];
      assert(recv_indx[i] < recv_size);
    }
  }

  Vector<Type> send_buff;
  {  // Prepare send buffer
    send_buff.ReInit(send_size * data_dim);
    ConstIterator<Type> data = data_.begin();
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < send_size; i++) {
      Long src_indx = psorted[i].data * data_dim;
      Long trg_indx = i * data_dim;
      for (Long j = 0; j < data_dim; j++) send_buff[trg_indx + j] = data[src_indx + j];
    }
  }

  Vector<Type> recv_buff;
  {  // All2Allv
    recv_buff.ReInit(recv_size * data_dim);
#pragma omp parallel for schedule(static)
    for (Integer i = 0; i < npes; i++) {
      sendSz[i] *= data_dim;
      sendOff[i] *= data_dim;
      recvSz[i] *= data_dim;
      recvOff[i] *= data_dim;
    }
    Alltoallv(send_buff.begin(), sendSz.begin(), sendOff.begin(), recv_buff.begin(), recvSz.begin(), recvOff.begin());
  }

  {  // Build output data.
    data_.ReInit(recv_size * data_dim);
    Iterator<Type> data = data_.begin();
#pragma omp parallel for schedule(static)
    for (Long i = 0; i < recv_size; i++) {
      Long src_indx = i * data_dim;
      Long trg_indx = recv_indx[i] * data_dim;
      for (Long j = 0; j < data_dim; j++) data[trg_indx + j] = recv_buff[src_indx + j];
    }
  }
}

#ifdef SCTL_HAVE_MPI
inline Vector<MPI_Request>* Comm::NewReq() const {
  if (req.empty()) req.push(new Vector<MPI_Request>);
  Vector<MPI_Request>& request = *(Vector<MPI_Request>*)req.top();
  req.pop();
  return &request;
}

inline void Comm::Init(const MPI_Comm mpi_comm) {
  #pragma omp critical(SCTL_COMM_DUP)
  MPI_Comm_dup(mpi_comm, &mpi_comm_);
  MPI_Comm_rank(mpi_comm_, &mpi_rank_);
  MPI_Comm_size(mpi_comm_, &mpi_size_);
}

inline void Comm::DelReq(Vector<MPI_Request>* req_ptr) const {
  if (req_ptr) req.push(req_ptr);
}

#define SCTL_HS_MPIDATATYPE(CTYPE, MPITYPE)              \
  template <> class Comm::CommDatatype<CTYPE> {     \
   public:                                          \
    static MPI_Datatype value() { return MPITYPE; } \
    static MPI_Op sum() { return MPI_SUM; }         \
    static MPI_Op min() { return MPI_MIN; }         \
    static MPI_Op max() { return MPI_MAX; }         \
  }

SCTL_HS_MPIDATATYPE(short, MPI_SHORT);
SCTL_HS_MPIDATATYPE(int, MPI_INT);
SCTL_HS_MPIDATATYPE(long, MPI_LONG);
SCTL_HS_MPIDATATYPE(unsigned short, MPI_UNSIGNED_SHORT);
SCTL_HS_MPIDATATYPE(unsigned int, MPI_UNSIGNED);
SCTL_HS_MPIDATATYPE(unsigned long, MPI_UNSIGNED_LONG);
SCTL_HS_MPIDATATYPE(float, MPI_FLOAT);
SCTL_HS_MPIDATATYPE(double, MPI_DOUBLE);
SCTL_HS_MPIDATATYPE(long double, MPI_LONG_DOUBLE);
SCTL_HS_MPIDATATYPE(long long, MPI_LONG_LONG_INT);
SCTL_HS_MPIDATATYPE(char, MPI_CHAR);
SCTL_HS_MPIDATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
#undef SCTL_HS_MPIDATATYPE
#endif

template <class Type, class Compare> void Comm::HyperQuickSort(const Vector<Type>& arr_, Vector<Type>& SortedElem, Compare comp) const {  // O( ((N/p)+log(p))*(log(N/p)+log(p)) )
  static_assert(std::is_trivially_copyable<Type>::value, "Data is not trivially copyable!");
#ifdef SCTL_HAVE_MPI
  Integer npes, myrank, omp_p;
  {  // Get comm size and rank.
    npes = Size();
    myrank = Rank();
    omp_p = omp_get_max_threads();
  }
  srand(myrank);

  Long totSize;
  {                 // Local and global sizes. O(log p)
    Long nelem = arr_.Dim();
    Allreduce<Long>(Ptr2ConstItr<Long>(&nelem, 1), Ptr2Itr<Long>(&totSize, 1), 1, CommOp::SUM);
  }

  if (npes == 1) {  // SortedElem <--- local_sort(arr_)
    SortedElem = arr_;
    omp_par::merge_sort(SortedElem.begin(), SortedElem.end(), comp);
    return;
  }

  Vector<Type> arr;
  {  // arr <-- local_sort(arr_)
    arr = arr_;
    omp_par::merge_sort(arr.begin(), arr.end(), comp);
  }

  Vector<Type> nbuff, nbuff_ext, rbuff, rbuff_ext;  // Allocate memory.
  MPI_Comm comm = mpi_comm_;                        // Copy comm
  bool free_comm = false;                           // Flag to free comm.

  // Binary split and merge in each iteration.
  while (npes > 1 && totSize > 0) {  // O(log p) iterations.
    Type split_key;
    Long totSize_new;
    {  // Determine split_key. O( log(N/p) + log(p) )
      Integer glb_splt_count;
      Vector<Type> glb_splitters;
      {  // Take random splitters. glb_splt_count = const = 100~1000
        Integer splt_count;
        Long nelem = arr.Dim();
        {  // Set splt_coun. O( 1 ) -- Let p * splt_count = t
          splt_count = (100 * nelem) / totSize;
          if (npes > 100) splt_count = (drand48() * totSize) < (100 * nelem) ? 1 : 0;
          if (splt_count > nelem) splt_count = nelem;
          MPI_Allreduce  (&splt_count, &glb_splt_count, 1, CommDatatype<Integer>::value(), CommDatatype<Integer>::sum(), comm);
          if (!glb_splt_count) splt_count = std::min<Long>(1, nelem);
          MPI_Allreduce  (&splt_count, &glb_splt_count, 1, CommDatatype<Integer>::value(), CommDatatype<Integer>::sum(), comm);
          SCTL_ASSERT(glb_splt_count);
        }

        Vector<Type> splitters(splt_count);
        for (Integer i = 0; i < splt_count; i++) {
          splitters[i] = arr[rand() % nelem];
        }

        Vector<Integer> glb_splt_cnts(npes), glb_splt_disp(npes);
        {  // Set glb_splt_cnts, glb_splt_disp
          MPI_Allgather(&splt_count, 1, CommDatatype<Integer>::value(), &glb_splt_cnts[0], 1, CommDatatype<Integer>::value(), comm);
          glb_splt_disp[0] = 0;
          omp_par::scan(glb_splt_cnts.begin(), glb_splt_disp.begin(), npes);
          SCTL_ASSERT(glb_splt_count == glb_splt_cnts[npes - 1] + glb_splt_disp[npes - 1]);
        }

        {  // Gather all splitters. O( log(p) )
          glb_splitters.ReInit(glb_splt_count);
          Vector<int> glb_splt_cnts_(npes), glb_splt_disp_(npes);
          for (Integer i = 0; i < npes; i++) {
            glb_splt_cnts_[i] = glb_splt_cnts[i];
            glb_splt_disp_[i] = glb_splt_disp[i];
          }
          MPI_Allgatherv((splt_count ? &splitters[0] : nullptr), splt_count, CommDatatype<Type>::value(), &glb_splitters[0], &glb_splt_cnts_[0], &glb_splt_disp_[0], CommDatatype<Type>::value(), comm);
        }
      }

      // Determine split key. O( log(N/p) + log(p) )
      Vector<Long> lrank(glb_splt_count);
      {  // Compute local rank
#pragma omp parallel for schedule(static)
        for (Integer i = 0; i < glb_splt_count; i++) {
          lrank[i] = std::lower_bound(arr.begin(), arr.end(), glb_splitters[i], comp) - arr.begin();
        }
      }

      Vector<Long> grank(glb_splt_count);
      {  // Compute global rank
        MPI_Allreduce(&lrank[0], &grank[0], glb_splt_count, CommDatatype<Long>::value(), CommDatatype<Long>::sum(), comm);
      }

      {  // Determine split_key, totSize_new
        Integer splitter_idx = 0;
        for (Integer i = 0; i < glb_splt_count; i++) {
          if (labs(grank[i] - totSize / 2) < labs(grank[splitter_idx] - totSize / 2)) {
            splitter_idx = i;
          }
        }
        split_key = glb_splitters[splitter_idx];

        if (myrank <= (npes - 1) / 2)
          totSize_new = grank[splitter_idx];
        else
          totSize_new = totSize - grank[splitter_idx];

        // double err=(((double)grank[splitter_idx])/(totSize/2))-1.0;
        // if(fabs<double>(err)<0.01 || npes<=16) break;
        // else if(!myrank) std::cout<<err<<'\n';
      }
    }

    Integer split_id = (npes - 1) / 2;
    {  // Split problem into two. O( N/p )
      Integer partner;
      {  // Set partner
        partner = myrank + (split_id+1) * (myrank<=split_id ? 1 : -1);
        if (partner >= npes) partner = npes - 1;
        assert(partner >= 0);
      }
      bool extra_partner = (npes % 2 == 1 && myrank == npes - 1);

      Long ssize = 0, lsize = 0;
      ConstIterator<Type> sbuff, lbuff;
      {  // Set ssize, lsize, sbuff, lbuff
        Long split_indx = std::lower_bound(arr.begin(), arr.end(), split_key, comp) - arr.begin();
        ssize = (myrank > split_id ? split_indx : arr.Dim() - split_indx);
        sbuff = (myrank > split_id ? arr.begin() : arr.begin() + split_indx);
        lsize = (myrank <= split_id ? split_indx : arr.Dim() - split_indx);
        lbuff = (myrank <= split_id ? arr.begin() : arr.begin() + split_indx);
      }

      Long rsize = 0, ext_rsize = 0;
      {  // Get rsize, ext_rsize
        Long ext_ssize = 0;
        MPI_Status status;
        MPI_Sendrecv(&ssize, 1, CommDatatype<Long>::value(), partner, 0, &rsize, 1, CommDatatype<Long>::value(), partner, 0, comm, &status);
        if (extra_partner) MPI_Sendrecv(&ext_ssize, 1, CommDatatype<Long>::value(), split_id, 0, &ext_rsize, 1, CommDatatype<Long>::value(), split_id, 0, comm, &status);
      }

      {  // Exchange data.
        rbuff.ReInit(rsize);
        rbuff_ext.ReInit(ext_rsize);
        MPI_Status status;
        MPI_Sendrecv((ssize ? &sbuff[0] : nullptr), ssize, CommDatatype<Type>::value(), partner, 0, (rsize ? &rbuff[0] : nullptr), rsize, CommDatatype<Type>::value(), partner, 0, comm, &status);
        if (extra_partner) MPI_Sendrecv(nullptr, 0, CommDatatype<Type>::value(), split_id, 0, (ext_rsize ? &rbuff_ext[0] : nullptr), ext_rsize, CommDatatype<Type>::value(), split_id, 0, comm, &status);
      }

      {  // nbuff <-- merge(lbuff, rbuff, rbuff_ext)
        nbuff.ReInit(lsize + rsize);
        omp_par::merge<ConstIterator<Type>>(lbuff, (lbuff + lsize), rbuff.begin(), rbuff.begin() + rsize, nbuff.begin(), omp_p, comp);
        if (ext_rsize > 0) {
          if (nbuff.Dim() > 0) {
            nbuff_ext.ReInit(lsize + rsize + ext_rsize);
            omp_par::merge(nbuff.begin(), nbuff.begin() + (lsize + rsize), rbuff_ext.begin(), rbuff_ext.begin() + ext_rsize, nbuff_ext.begin(), omp_p, comp);
            nbuff.Swap(nbuff_ext);
            nbuff_ext.ReInit(0);
          } else {
            nbuff.Swap(rbuff_ext);
          }
        }
      }

      // Copy new data.
      totSize = totSize_new;
      arr.Swap(nbuff);
    }

    {  // Split comm.  O( log(p) ) ??
      MPI_Comm scomm;
      #pragma omp critical(SCTL_COMM_DUP)
      MPI_Comm_split(comm, myrank <= split_id, myrank, &scomm);
      #pragma omp critical(SCTL_COMM_DUP)
      if (free_comm) MPI_Comm_free(&comm);
      comm = scomm;
      free_comm = true;

      npes = (myrank <= split_id ? split_id + 1 : npes - split_id - 1);
      myrank = (myrank <= split_id ? myrank : myrank - split_id - 1);
    }
  }
  #pragma omp critical(SCTL_COMM_DUP)
  if (free_comm) MPI_Comm_free(&comm);

  SortedElem = arr;
  PartitionW<Type>(SortedElem);
#else
  SortedElem = arr_;
  std::sort(SortedElem.begin(), SortedElem.begin() + SortedElem.Dim(), comp);
#endif
}

}  // end namespace
