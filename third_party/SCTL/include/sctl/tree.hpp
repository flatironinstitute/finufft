#ifndef _SCTL_TREE_
#define _SCTL_TREE_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(morton.hpp)
#include SCTL_INCLUDE(vtudata.hpp)
#include SCTL_INCLUDE(ompUtils.hpp)

#include <string>
#include <vector>
#include <algorithm>

namespace SCTL_NAMESPACE {

template <Integer DIM> class Tree {
  public:

    struct NodeAttr {
      unsigned char Leaf : 1, Ghost : 1;
    };

    struct NodeLists {
      Long p2n;
      Long parent;
      Long child[1 << DIM];
      Long nbr[sctl::pow<DIM,Integer>(3)];
    };

    static constexpr Integer Dim() {
      return DIM;
    }

    Tree(const Comm& comm_ = Comm::Self()) : comm(comm_) {
      Integer rank = comm.Rank();
      Integer np = comm.Size();

      Vector<double> coord;
      { // Set coord
        Long N0 = 1;
        while (sctl::pow<DIM,Long>(N0) < np) N0++;
        Long N = sctl::pow<DIM,Long>(N0);
        Long start = N * (rank+0) / np;
        Long end   = N * (rank+1) / np;
        coord.ReInit((end-start)*DIM);
        for (Long i = start; i < end; i++) {
          Long  idx = i;
          for (Integer k = 0; k < DIM; k++) {
            coord[(i-start)*DIM+k] = (idx % N0) / (double)N0;
            idx /= N0;
          }
        }
      }
      this->UpdateRefinement(coord);
    }

    ~Tree() {
      #ifdef SCTL_MEMDEBUG
      for (auto& pair : node_data) {
        SCTL_ASSERT(node_cnt.find(pair.first) != node_cnt.end());
      }
      #endif
    }

    const Vector<Morton<DIM>>& GetPartitionMID() const {
      return mins;
    }
    const Vector<Morton<DIM>>& GetNodeMID() const {
      return node_mid;
    }
    const Vector<NodeAttr>& GetNodeAttr() const {
      return node_attr;
    }
    const Vector<NodeLists>& GetNodeLists() const {
      return node_lst;
    }
    const Comm& GetComm() const {
      return comm;
    }

    template <class Real> void UpdateRefinement(const Vector<Real>& coord, Long M = 1, bool balance21 = 0, bool periodic = 0) {
      Integer np = comm.Size();
      Integer rank = comm.Rank();

      Vector<Morton<DIM>> node_mid_orig;
      Long start_idx_orig, end_idx_orig;
      if (mins.Dim()) { // Set start_idx_orig, end_idx_orig
        start_idx_orig = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
        end_idx_orig = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
        node_mid_orig.ReInit(end_idx_orig - start_idx_orig, node_mid.begin() + start_idx_orig, true);
      } else {
        start_idx_orig = 0;
        end_idx_orig = 0;
      }

      auto coarsest_ancestor_mid = [](const Morton<DIM>& m0) {
        Morton<DIM> md;
        Integer d0 = m0.Depth();
        for (Integer d = 0; d <= d0; d++) {
          md = m0.Ancestor(d);
          if (md.Ancestor(d0) == m0) break;
        }
        return md;
      };

      Morton<DIM> pt_mid0;
      Vector<Morton<DIM>> pt_mid;
      { // Construct sorted pt_mid
        Long Npt = coord.Dim() / DIM;
        pt_mid.ReInit(Npt);
        for (Long i = 0; i < Npt; i++) {
          pt_mid[i] = Morton<DIM>(coord.begin() + i*DIM);
        }
        Vector<Morton<DIM>> sorted_mid;
        comm.HyperQuickSort(pt_mid, sorted_mid);
        pt_mid.Swap(sorted_mid);
        SCTL_ASSERT(pt_mid.Dim());
        pt_mid0 = pt_mid[0];
      }
      { // Update M = global_min(pt_mid.Dim(), M)
        Long M0, M1, Npt = pt_mid.Dim();
        comm.Allreduce(Ptr2ConstItr<Long>(&M,1), Ptr2Itr<Long>(&M0,1), 1, Comm::CommOp::MIN);
        comm.Allreduce(Ptr2ConstItr<Long>(&Npt,1), Ptr2Itr<Long>(&M1,1), 1, Comm::CommOp::MIN);
        M = std::min(M0,M1);
        SCTL_ASSERT(M > 0);
      }
      { // pt_mid <-- [M points from rank-1; pt_mid; M points from rank+1]
        Long send_size0 = (rank+1<np ? M : 0);
        Long send_size1 = (rank  > 0 ? M : 0);
        Long recv_size0 = (rank  > 0 ? M : 0);
        Long recv_size1 = (rank+1<np ? M : 0);
        Vector<Morton<DIM>> pt_mid_(recv_size0 + pt_mid.Dim() + recv_size1);
        memcopy(pt_mid_.begin()+recv_size0, pt_mid.begin(), pt_mid.Dim());

        void* recv_req0 = comm.Irecv(pt_mid_.begin(), recv_size0, (rank+np-1)%np, 0);
        void* recv_req1 = comm.Irecv(pt_mid_.begin() + recv_size0 + pt_mid.Dim(), recv_size1, (rank+1)%np, 1);
        void* send_req0 = comm.Isend(pt_mid .begin() + pt_mid.Dim() - send_size0, send_size0, (rank+1)%np, 0);
        void* send_req1 = comm.Isend(pt_mid .begin(), send_size1, (rank+np-1)%np, 1);
        comm.Wait(recv_req0);
        comm.Wait(recv_req1);
        comm.Wait(send_req0);
        comm.Wait(send_req1);
        pt_mid.Swap(pt_mid_);
      }
      { // Build linear MortonID tree from pt_mid
        node_mid.ReInit(0);
        Long idx = 0;
        Morton<DIM> m0;
        Morton<DIM> mend = Morton<DIM>().Next();
        while (m0 < mend) {
          Integer d = m0.Depth();
          Morton<DIM> m1 = (idx + M < pt_mid.Dim() ? pt_mid[idx+M] : Morton<DIM>().Next());
          while (d < Morton<DIM>::MAX_DEPTH && m0.Ancestor(d) == m1.Ancestor(d)) {
            node_mid.PushBack(m0.Ancestor(d));
            d++;
          }
          m0 = m0.Ancestor(d);
          node_mid.PushBack(m0);
          m0 = m0.Next();
          idx = std::lower_bound(pt_mid.begin(), pt_mid.end(), m0) - pt_mid.begin();
        }
      }
      { // Set mins
        mins.ReInit(np);
        Long min_idx = std::lower_bound(node_mid.begin(), node_mid.end(), pt_mid0) - node_mid.begin() - 1;
        if (!rank || min_idx < 0) min_idx = 0;
        Morton<DIM> m0 = coarsest_ancestor_mid(node_mid[min_idx]);
        comm.Allgather(Ptr2ConstItr<Morton<DIM>>(&m0,1), 1, mins.begin(), 1);
      }
      if (balance21) { // 2:1 balance refinement // TODO: optimize
        Vector<Morton<DIM>> parent_mid;
        { // add balancing Morton IDs
          Vector<std::set<Morton<DIM>>> parent_mid_set(Morton<DIM>::MAX_DEPTH+1);
          Vector<Morton<DIM>> nlst;
          for (const auto& m0 : node_mid) {
            Integer d0 = m0.Depth();
            parent_mid_set[m0.Depth()].insert(m0.Ancestor(d0-1));
          }
          for (Integer d = Morton<DIM>::MAX_DEPTH; d > 0; d--) {
            for (const auto& m : parent_mid_set[d]) {
              m.NbrList(nlst, d-1, periodic);
              parent_mid_set[d-1].insert(nlst.begin(), nlst.end());
              parent_mid.PushBack(m);
            }
          }
        }

        Vector<Morton<DIM>> parent_mid_sorted;
        { // sort and repartition
          comm.HyperQuickSort(parent_mid, parent_mid_sorted);
          comm.PartitionS(parent_mid_sorted, mins[comm.Rank()]);
        }

        Vector<Morton<DIM>> tmp_mid;
        { // add children
          Vector<Morton<DIM>> clst;
          tmp_mid.PushBack(Morton<DIM>()); // include root node
          for (Long i = 0; i < parent_mid_sorted.Dim(); i++) {
            if (i+1 == parent_mid_sorted.Dim() || parent_mid_sorted[i] != parent_mid_sorted[i+1]) {
              const auto& m = parent_mid_sorted[i];
              tmp_mid.PushBack(m);
              m.Children(clst);
              for (const auto& c : clst) tmp_mid.PushBack(c);
            }
          }
          auto insert_ancestor_children = [](Vector<Morton<DIM>>& mvec, const Morton<DIM>& m0) {
            Integer d0 = m0.Depth();
            Vector<Morton<DIM>> clst;
            for (Integer d = 0; d < d0; d++) {
              m0.Ancestor(d).Children(clst);
              for (const auto& m : clst) mvec.PushBack(m);
            }
          };
          insert_ancestor_children(tmp_mid, mins[rank]);
          omp_par::merge_sort(tmp_mid.begin(), tmp_mid.end());
        }

        node_mid.ReInit(0);
        for (Long i = 0; i < tmp_mid.Dim(); i++) { // remove duplicates
          if (i+1 == tmp_mid.Dim() || tmp_mid[i] != tmp_mid[i+1]) {
            node_mid.PushBack(tmp_mid[i]);
          }
        }
      }
      { // Add place-holder for ghost nodes
        Long start_idx, end_idx;
        { // Set start_idx, end_idx
          start_idx = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
          end_idx = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
        }
        { // Set user_mid, user_cnt
          Vector<SortPair<Long,Morton<DIM>>> user_node_lst;
          Vector<Morton<DIM>> nlst;
          std::set<Long> user_procs;
          for (Long i = start_idx; i < end_idx; i++) {
            Morton<DIM> m0 = node_mid[i];
            Integer d0 = m0.Depth();
            m0.NbrList(nlst, std::max<Integer>(d0-2,0), periodic);
            user_procs.clear();
            for (const auto& m : nlst) {
              Morton<DIM> m_start = m.DFD();
              Morton<DIM> m_end = m.Next();
              Integer p_start = std::lower_bound(mins.begin(), mins.end(), m_start) - mins.begin() - 1;
              Integer p_end   = std::lower_bound(mins.begin(), mins.end(), m_end  ) - mins.begin();
              SCTL_ASSERT(0 <= p_start);
              SCTL_ASSERT(p_start < p_end);
              SCTL_ASSERT(p_end <= np);
              for (Long p = p_start; p < p_end; p++) {
                if (p != rank) user_procs.insert(p);
              }
            }
            for (const auto p : user_procs) {
              SortPair<Long,Morton<DIM>> pair;
              pair.key = p;
              pair.data = m0;
              user_node_lst.PushBack(pair);
            }
          }
          omp_par::merge_sort(user_node_lst.begin(), user_node_lst.end());

          user_cnt.ReInit(np);
          user_mid.ReInit(user_node_lst.Dim());
          for (Integer i = 0; i < np; i++) {
            SortPair<Long,Morton<DIM>> pair_start, pair_end;
            pair_start.key = i;
            pair_end.key = i+1;
            Long cnt_start = std::lower_bound(user_node_lst.begin(), user_node_lst.end(), pair_start) - user_node_lst.begin();
            Long cnt_end   = std::lower_bound(user_node_lst.begin(), user_node_lst.end(), pair_end  ) - user_node_lst.begin();
            user_cnt[i] = cnt_end - cnt_start;
            for (Long j = cnt_start; j < cnt_end; j++) {
              user_mid[j] = user_node_lst[j].data;
            }
            std::sort(user_mid.begin() + cnt_start, user_mid.begin() + cnt_end);
          }
        }

        Vector<Morton<DIM>> ghost_mid;
        { // SendRecv user_mid
          const Vector<Long>& send_cnt = user_cnt;
          Vector<Long> send_dsp(np);
          scan(send_dsp, send_cnt);

          Vector<Long> recv_cnt(np), recv_dsp(np);
          comm.Alltoall(send_cnt.begin(), 1, recv_cnt.begin(), 1);
          scan(recv_dsp, recv_cnt);

          const Vector<Morton<DIM>>& send_mid = user_mid;
          Long Nsend = send_dsp[np-1] + send_cnt[np-1];
          Long Nrecv = recv_dsp[np-1] + recv_cnt[np-1];
          SCTL_ASSERT(send_mid.Dim() == Nsend);

          ghost_mid.ReInit(Nrecv);
          comm.Alltoallv(send_mid.begin(), send_cnt.begin(), send_dsp.begin(), ghost_mid.begin(), recv_cnt.begin(), recv_dsp.begin());
        }

        { // Update node_mid <-- ghost_mid + node_mid
          Vector<Morton<DIM>> new_mid(end_idx-start_idx + ghost_mid.Dim());
          Long Nsplit = std::lower_bound(ghost_mid.begin(), ghost_mid.end(), mins[rank]) - ghost_mid.begin();
          for (Long i = 0; i < Nsplit; i++) {
            new_mid[i] = ghost_mid[i];
          }
          for (Long i = 0; i < end_idx - start_idx; i++) {
            new_mid[Nsplit + i] = node_mid[start_idx + i];
          }
          for (Long i = Nsplit; i < ghost_mid.Dim(); i++) {
            new_mid[end_idx - start_idx + i] = ghost_mid[i];
          }
          node_mid.Swap(new_mid);
        }
      }
      { // Set node_mid, node_attr
        Morton<DIM> m0 = (rank      ? mins[rank]   : Morton<DIM>()       );
        Morton<DIM> m1 = (rank+1<np ? mins[rank+1] : Morton<DIM>().Next());
        Long Nnodes = node_mid.Dim();
        node_attr.ReInit(Nnodes);
        for (Long i = 0; i < Nnodes; i++) {
          node_attr[i].Leaf = !(i+1<Nnodes && node_mid[i].isAncestor(node_mid[i+1]));
          node_attr[i].Ghost = (node_mid[i] < m0 || node_mid[i] >= m1);
        }
      }
      { // Set node_lst
        static constexpr Integer MAX_CHILD = (1u << DIM);
        static constexpr Integer MAX_NBRS = sctl::pow<DIM,Integer>(3);
        Long Nnodes = node_mid.Dim();
        node_lst.ReInit(Nnodes);

        Vector<Long> ancestors(Morton<DIM>::MAX_DEPTH);
        Vector<Long> child_cnt(Morton<DIM>::MAX_DEPTH);
        #pragma omp parallel for schedule(static)
        for (Long i = 0; i < Nnodes; i++) {
          node_lst[i].p2n = -1;
          node_lst[i].parent = -1;
          for (Integer j = 0; j < MAX_CHILD; j++) node_lst[i].child[j] = -1;
          for (Integer j = 0; j < MAX_NBRS; j++) node_lst[i].nbr[j] = -1;
        }
        for (Long i = 0; i < Nnodes; i++) { // Set parent_lst, child_lst_
          Integer depth = node_mid[i].Depth();
          ancestors[depth] = i;
          child_cnt[depth] = 0;
          if (depth) {
            Long p = ancestors[depth-1];
            Long& c = child_cnt[depth-1];
            node_lst[i].parent = p;
            node_lst[p].child[c] = i;
            node_lst[p].p2n = c;
            c++;
          }
        }
        // TODO: add nbr-list
      }
      if (0) { // Check tree
        Morton<DIM> m0;
        SCTL_ASSERT(node_mid.Dim() && m0 == node_mid[0]);
        for (Long i = 1; i < node_mid.Dim(); i++) {
          const auto& m = node_mid[i];
          if (m0.isAncestor(m)) m0 = m0.Ancestor(m0.Depth()+1);
          else m0 = m0.Next();
          SCTL_ASSERT(m0 == m);
        }
        SCTL_ASSERT(m0.Next() == Morton<DIM>().Next());
      }

      { // Update node_data, node_cnt
        Long start_idx, end_idx;
        { // Set start_idx, end_idx
          start_idx = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
          end_idx = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
        }

        comm.PartitionS(node_mid_orig, mins[comm.Rank()]);

        Vector<Long> new_cnt_range0(node_mid.Dim()), new_cnt_range1(node_mid.Dim());
        { // Set new_cnt_range0, new_cnt_range1
          for (Long i = 0; i < start_idx; i++) {
            new_cnt_range0[i] = 0;
            new_cnt_range1[i] = 0;
          }
          for (Long i = start_idx; i < end_idx; i++) {
            auto m0 = (node_mid[i+0]);
            auto m1 = (i+1==end_idx ? Morton<DIM>().Next() : (node_mid[i+1]));
            new_cnt_range0[i] = std::lower_bound(node_mid_orig.begin(), node_mid_orig.begin() + node_mid_orig.Dim(), m0) - node_mid_orig.begin();
            new_cnt_range1[i] = std::lower_bound(node_mid_orig.begin(), node_mid_orig.begin() + node_mid_orig.Dim(), m1) - node_mid_orig.begin();
          }
          for (Long i = end_idx; i < node_mid.Dim(); i++) {
            new_cnt_range0[i] = 0;
            new_cnt_range1[i] = 0;
          }
        }

        Vector<Long> cnt_tmp;
        Vector<char> data_tmp;
        for (const auto& pair : node_data) {
          const std::string& data_name = pair.first;

          Long dof;
          Iterator<Vector<char>> data_;
          Iterator<Vector<Long>> cnt_;
          GetData_(data_, cnt_, data_name);
          { // Set dof
            StaticArray<Long,2> Nl, Ng;
            Nl[0] = data_->Dim();
            Nl[1] = omp_par::reduce(cnt_->begin(), cnt_->Dim());
            comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, Comm::CommOp::SUM);
            dof = Ng[0] / std::max<Long>(Ng[1],1);
            SCTL_ASSERT(Nl[0] == Nl[1] * dof);
            SCTL_ASSERT(Ng[0] == Ng[1] * dof);
          }

          Long data_dsp = omp_par::reduce(cnt_->begin(), start_idx_orig);
          Long data_cnt = omp_par::reduce(cnt_->begin() + start_idx_orig, end_idx_orig - start_idx_orig);
          data_tmp.ReInit(data_cnt * dof, data_->begin() + data_dsp * dof, true);

          cnt_tmp.ReInit(end_idx_orig - start_idx_orig, cnt_->begin() + start_idx_orig, true);
          comm.PartitionN(cnt_tmp, node_mid_orig.Dim());

          cnt_->ReInit(node_mid.Dim());
          for (Long i = 0; i < node_mid.Dim(); i++) {
            Long sum = 0;
            Long j0 = new_cnt_range0[i];
            Long j1 = new_cnt_range1[i];
            for (Long j = j0; j < j1; j++) sum += cnt_tmp[j];
            cnt_[0][i] = sum;
          }
          SCTL_ASSERT(omp_par::reduce(cnt_->begin(), cnt_->Dim()) == omp_par::reduce(cnt_tmp.begin(), cnt_tmp.Dim()));

          Long Ndata = omp_par::reduce(cnt_->begin(), cnt_->Dim()) * dof;
          comm.PartitionN(data_tmp, Ndata);
          SCTL_ASSERT(data_tmp.Dim() == Ndata);
          data_->Swap(data_tmp);
        }
      }
    }

    template <class ValueType> void AddData(const std::string& name, const Vector<ValueType>& data, const Vector<Long>& cnt) {
      Long dof;
      { // Check dof
        StaticArray<Long,2> Nl, Ng;
        Nl[0] = data.Dim();
        Nl[1] = omp_par::reduce(cnt.begin(), cnt.Dim());
        comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, Comm::CommOp::SUM);
        dof = Ng[0] / std::max<Long>(Ng[1],1);
        SCTL_ASSERT(Nl[0] == Nl[1] * dof);
        SCTL_ASSERT(Ng[0] == Ng[1] * dof);
      }
      if (dof) SCTL_ASSERT(cnt.Dim() == node_mid.Dim());

      SCTL_ASSERT(node_data.find(name) == node_data.end());
      node_data[name].ReInit(data.Dim()*sizeof(ValueType), (Iterator<char>)data.begin(), true);
      node_cnt [name] = cnt;
    }

    template <class ValueType> void GetData(Vector<ValueType>& data, Vector<Long>& cnt, const std::string& name) const {
      const auto data_ = node_data.find(name);
      const auto cnt_ = node_cnt.find(name);
      SCTL_ASSERT(data_ != node_data.end());
      SCTL_ASSERT( cnt_ != node_cnt .end());
      data.ReInit(data_->second.Dim()/sizeof(ValueType), (Iterator<ValueType>)data_->second.begin(), false);
      SCTL_ASSERT(data.Dim()*(Long)sizeof(ValueType) == data_->second.Dim());
      cnt .ReInit( cnt_->second.Dim(), (Iterator<Long>)cnt_->second.begin(), false);
    }

    template <class ValueType> void ReduceBroadcast(const std::string& name) {
      Integer np = comm.Size();
      Integer rank = comm.Rank();

      Vector<Long> dsp;
      Iterator<Vector<char>> data_;
      Iterator<Vector<Long>> cnt_;
      GetData_(data_, cnt_, name);
      Vector<ValueType> data(data_->Dim()/sizeof(ValueType), (Iterator<ValueType>)data_->begin(), false);
      Vector<Long>& cnt = *cnt_;
      scan(dsp, cnt);

      Long dof;
      { // Set dof
        StaticArray<Long,2> Nl, Ng;
        Nl[0] = data.Dim();
        Nl[1] = omp_par::reduce(cnt.begin(), cnt.Dim());
        comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, Comm::CommOp::SUM);
        dof = Ng[0] / std::max<Long>(Ng[1],1);
        SCTL_ASSERT(Nl[0] == Nl[1] * dof);
        SCTL_ASSERT(Ng[0] == Ng[1] * dof);
      }

      { // Reduce
        Vector<Morton<DIM>> send_mid, recv_mid;
        Vector<Long> send_node_cnt(np), send_node_dsp(np);
        Vector<Long> recv_node_cnt(np), recv_node_dsp(np);
        { // Set send_mid, send_node_cnt, send_node_dsp, recv_mid, recv_node_cnt, recv_node_dsp
          { // Set send_mid
            Morton<DIM> m0 = mins[rank];
            for (Integer d = 0; d < m0.Depth(); d++) {
              send_mid.PushBack(m0.Ancestor(d));
            }
          }
          for (Integer p = 0; p < np; p++) {
            Long start_idx = std::lower_bound(send_mid.begin(), send_mid.end(), mins[p]) - send_mid.begin();
            Long end_idx = std::lower_bound(send_mid.begin(), send_mid.end(), (p+1==np ? Morton<DIM>().Next() : mins[p+1])) - send_mid.begin();
            send_node_cnt[p] = end_idx - start_idx;
          }
          scan(send_node_dsp, send_node_cnt);
          SCTL_ASSERT(send_node_dsp[np-1]+send_node_cnt[np-1] == send_mid.Dim());
          comm.Alltoall(send_node_cnt.begin(), 1, recv_node_cnt.begin(), 1);
          scan(recv_node_dsp, recv_node_cnt);

          recv_mid.ReInit(recv_node_dsp[np-1] + recv_node_cnt[np-1]);
          comm.Alltoallv(send_mid.begin(), send_node_cnt.begin(), send_node_dsp.begin(), recv_mid.begin(), recv_node_cnt.begin(), recv_node_dsp.begin());
        }

        Vector<Long> send_data_cnt, send_data_dsp;
        Vector<Long> recv_data_cnt, recv_data_dsp;
        { // Set send_data_cnt, send_data_dsp
          send_data_cnt.ReInit(send_mid.Dim());
          recv_data_cnt.ReInit(recv_mid.Dim());
          for (Long i = 0; i < send_mid.Dim(); i++) {
            Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), send_mid[i]) - node_mid.begin();
            SCTL_ASSERT(send_mid[i] == node_mid[idx]);
            send_data_cnt[i] = cnt[idx];
          }
          scan(send_data_dsp, send_data_cnt);
          comm.Alltoallv(send_data_cnt.begin(), send_node_cnt.begin(), send_node_dsp.begin(), recv_data_cnt.begin(), recv_node_cnt.begin(), recv_node_dsp.begin());
          scan(recv_data_dsp, recv_data_cnt);
        }

        Vector<ValueType> send_buff, recv_buff;
        Vector<Long> send_buff_cnt(np), send_buff_dsp(np);
        Vector<Long> recv_buff_cnt(np), recv_buff_dsp(np);
        { // Set send_buff, send_buff_cnt, send_buff_dsp, recv_buff, recv_buff_cnt, recv_buff_dsp
          Long N_send_nodes = send_mid.Dim();
          Long N_recv_nodes = recv_mid.Dim();
          if (N_send_nodes) send_buff.ReInit((send_data_dsp[N_send_nodes-1] + send_data_cnt[N_send_nodes-1]) * dof);
          if (N_recv_nodes) recv_buff.ReInit((recv_data_dsp[N_recv_nodes-1] + recv_data_cnt[N_recv_nodes-1]) * dof);
          for (Long i = 0; i < N_send_nodes; i++) {
            Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), send_mid[i]) - node_mid.begin();
            SCTL_ASSERT(send_mid[i] == node_mid[idx]);
            Long dsp_ = dsp[idx] * dof;
            Long cnt_ = cnt[idx] * dof;
            Long send_data_dsp_ = send_data_dsp[i] * dof;
            Long send_data_cnt_ = send_data_cnt[i] * dof;
            SCTL_ASSERT(send_data_cnt_ == cnt_);
            for (Long j = 0; j < cnt_; j++) {
              send_buff[send_data_dsp_+j] = data[dsp_+j];
            }
          }
          for (Integer p = 0; p < np; p++) {
            Long send_buff_cnt_ = 0;
            Long recv_buff_cnt_ = 0;
            for (Long i = 0; i < send_node_cnt[p]; i++) {
              send_buff_cnt_ += send_data_cnt[send_node_dsp[p]+i];
            }
            for (Long i = 0; i < recv_node_cnt[p]; i++) {
              recv_buff_cnt_ += recv_data_cnt[recv_node_dsp[p]+i];
            }
            send_buff_cnt[p] = send_buff_cnt_ * dof;
            recv_buff_cnt[p] = recv_buff_cnt_ * dof;
          }
          scan(send_buff_dsp, send_buff_cnt);
          scan(recv_buff_dsp, recv_buff_cnt);
          comm.Alltoallv(send_buff.begin(), send_buff_cnt.begin(), send_buff_dsp.begin(), recv_buff.begin(), recv_buff_cnt.begin(), recv_buff_dsp.begin());
        }

        { // Reduction
          Long N_recv_nodes = recv_mid.Dim();
          for (Long i = 0; i < N_recv_nodes; i++) {
            Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), recv_mid[i]) - node_mid.begin();
            Long dsp_ = dsp[idx] * dof;
            Long cnt_ = cnt[idx] * dof;
            Long recv_data_dsp_ = recv_data_dsp[i] * dof;
            Long recv_data_cnt_ = recv_data_cnt[i] * dof;
            if (recv_data_cnt_ == cnt_) {
              for (Long j = 0; j < cnt_; j++) {
                data[dsp_+j] += recv_buff[recv_data_dsp_+j];
              }
            }
          }
        }
      }

      Broadcast<ValueType>(name);
    }

    template <class ValueType> void Broadcast(const std::string& name) {
      Integer np = comm.Size();
      Integer rank = comm.Rank();

      Vector<Long> dsp;
      Iterator<Vector<char>> data_;
      Iterator<Vector<Long>> cnt_;
      GetData_(data_, cnt_, name);
      Vector<ValueType> data(data_->Dim()/sizeof(ValueType), (Iterator<ValueType>)data_->begin(), false);
      Vector<Long>& cnt = *cnt_;
      scan(dsp, cnt);

      Long dof;
      { // Set dof
        StaticArray<Long,2> Nl, Ng;
        Nl[0] = data.Dim();
        Nl[1] = omp_par::reduce(cnt.begin(), cnt.Dim());
        comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, Comm::CommOp::SUM);
        dof = Ng[0] / std::max<Long>(Ng[1],1);
        SCTL_ASSERT(Nl[0] == Nl[1] * dof);
        SCTL_ASSERT(Ng[0] == Ng[1] * dof);
      }

      { // Broadcast
        const Vector<Morton<DIM>>& send_mid = user_mid;
        const Vector<Long>& send_node_cnt = user_cnt;
        Vector<Long> send_node_dsp(np);
        { // Set send_dsp
          SCTL_ASSERT(send_node_cnt.Dim() == np);
          scan(send_node_dsp, send_node_cnt);
          SCTL_ASSERT(send_node_dsp[np-1] + send_node_cnt[np-1] == send_mid.Dim());
        }

        Vector<Morton<DIM>> recv_mid;
        Vector<Long> recv_node_cnt(np), recv_node_dsp(np);
        { // Set recv_mid, recv_node_cnt, recv_node_dsp
          comm.Alltoall(send_node_cnt.begin(), 1, recv_node_cnt.begin(), 1);
          scan(recv_node_dsp, recv_node_cnt);

          recv_mid.ReInit(recv_node_dsp[np-1] + recv_node_cnt[np-1]);
          comm.Alltoallv(send_mid.begin(), send_node_cnt.begin(), send_node_dsp.begin(), recv_mid.begin(), recv_node_cnt.begin(), recv_node_dsp.begin());
        }

        Vector<Long> send_data_cnt, send_data_dsp;
        Vector<Long> recv_data_cnt, recv_data_dsp;
        { // Set send_data_cnt, send_data_dsp
          send_data_cnt.ReInit(send_mid.Dim());
          recv_data_cnt.ReInit(recv_mid.Dim());
          for (Long i = 0; i < send_mid.Dim(); i++) {
            Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), send_mid[i]) - node_mid.begin();
            SCTL_ASSERT(send_mid[i] == node_mid[idx]);
            send_data_cnt[i] = cnt[idx];
          }
          scan(send_data_dsp, send_data_cnt);
          comm.Alltoallv(send_data_cnt.begin(), send_node_cnt.begin(), send_node_dsp.begin(), recv_data_cnt.begin(), recv_node_cnt.begin(), recv_node_dsp.begin());
          scan(recv_data_dsp, recv_data_cnt);
        }

        Vector<ValueType> send_buff, recv_buff;
        Vector<Long> send_buff_cnt(np), send_buff_dsp(np);
        Vector<Long> recv_buff_cnt(np), recv_buff_dsp(np);
        { // Set send_buff, send_buff_cnt, send_buff_dsp, recv_buff, recv_buff_cnt, recv_buff_dsp
          Long N_send_nodes = send_mid.Dim();
          Long N_recv_nodes = recv_mid.Dim();
          if (N_send_nodes) send_buff.ReInit((send_data_dsp[N_send_nodes-1] + send_data_cnt[N_send_nodes-1]) * dof);
          if (N_recv_nodes) recv_buff.ReInit((recv_data_dsp[N_recv_nodes-1] + recv_data_cnt[N_recv_nodes-1]) * dof);
          for (Long i = 0; i < N_send_nodes; i++) {
            Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), send_mid[i]) - node_mid.begin();
            SCTL_ASSERT(send_mid[i] == node_mid[idx]);
            Long dsp_ = dsp[idx] * dof;
            Long cnt_ = cnt[idx] * dof;
            Long send_data_dsp_ = send_data_dsp[i] * dof;
            Long send_data_cnt_ = send_data_cnt[i] * dof;
            SCTL_ASSERT(send_data_cnt_ == cnt_);
            for (Long j = 0; j < cnt_; j++) {
              send_buff[send_data_dsp_+j] = data[dsp_+j];
            }
          }
          for (Integer p = 0; p < np; p++) {
            Long send_buff_cnt_ = 0;
            Long recv_buff_cnt_ = 0;
            for (Long i = 0; i < send_node_cnt[p]; i++) {
              send_buff_cnt_ += send_data_cnt[send_node_dsp[p]+i];
            }
            for (Long i = 0; i < recv_node_cnt[p]; i++) {
              recv_buff_cnt_ += recv_data_cnt[recv_node_dsp[p]+i];
            }
            send_buff_cnt[p] = send_buff_cnt_ * dof;
            recv_buff_cnt[p] = recv_buff_cnt_ * dof;
          }
          scan(send_buff_dsp, send_buff_cnt);
          scan(recv_buff_dsp, recv_buff_cnt);
          comm.Alltoallv(send_buff.begin(), send_buff_cnt.begin(), send_buff_dsp.begin(), recv_buff.begin(), recv_buff_cnt.begin(), recv_buff_dsp.begin());
        }

        Long start_idx, end_idx;
        { // Set start_idx, end_idx
          start_idx = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
          end_idx = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
          SCTL_ASSERT(0 <= start_idx);
          SCTL_ASSERT(start_idx < end_idx);
          SCTL_ASSERT(end_idx <= node_mid.Dim());
        }

        { // Update data <-- data + recv_buff
          Long Nsplit = std::lower_bound(recv_mid.begin(), recv_mid.end(), mins[rank]) - recv_mid.begin();
          SCTL_ASSERT(recv_mid.Dim()-Nsplit == node_mid.Dim()-end_idx);
          SCTL_ASSERT(Nsplit == start_idx);

          Long N0 = (start_idx ? dsp[start_idx-1] + cnt[start_idx-1] : 0) * dof;
          Long N1 = (end_idx ? dsp[end_idx-1] + cnt[end_idx-1] : 0) * dof;
          Long Ns = (Nsplit ? recv_data_dsp[Nsplit-1] + recv_data_cnt[Nsplit-1] : 0) * dof;
          if (N0 != Ns || recv_buff.Dim() != N0+data.Dim()-N1) { // resize data and preserve non-ghost data
            Vector<char> data_new((recv_buff.Dim() + N1-N0) * sizeof(ValueType));
            memcopy(data_new.begin() + Ns * sizeof(ValueType), data_->begin() + N0 * sizeof(ValueType), (N1-N0) * sizeof(ValueType));
            data_->Swap(data_new);
            data.ReInit(data_->Dim()/sizeof(ValueType), (Iterator<ValueType>)data_->begin(), false);
          }

          memcopy(cnt.begin(), recv_data_cnt.begin(), start_idx);
          memcopy(cnt.begin()+end_idx, recv_data_cnt.begin()+Nsplit, node_mid.Dim()-end_idx);

          memcopy(data.begin(), recv_buff.begin(), Ns);
          memcopy(data.begin()+data.Dim()+Ns-recv_buff.Dim(), recv_buff.begin()+Ns, recv_buff.Dim()-Ns);
        }
      }
    }

    void DeleteData(const std::string& name) {
      SCTL_ASSERT(node_data.find(name) != node_data.end());
      SCTL_ASSERT(node_cnt .find(name) != node_cnt .end());
      node_data.erase(name);
      node_cnt .erase(name);
    }

    void WriteTreeVTK(std::string fname, bool show_ghost = false) const {
      typedef typename VTUData::VTKReal VTKReal;
      VTUData vtu_data;
      if (DIM <= 3) {  // Set vtu data
        static const Integer Ncorner = (1u << DIM);

        Vector<VTKReal> &coord = vtu_data.coord;
        //Vector<VTKReal> &value = vtu_data.value;

        Vector<int32_t> &connect = vtu_data.connect;
        Vector<int32_t> &offset = vtu_data.offset;
        Vector<uint8_t> &types = vtu_data.types;

        StaticArray<VTKReal, DIM> c;
        Long point_cnt = coord.Dim() / 3;
        Long connect_cnt = connect.Dim();
        for (Long nid = 0; nid < node_mid.Dim(); nid++) {
          const Morton<DIM> &mid = node_mid[nid];
          const NodeAttr &attr = node_attr[nid];
          if (!show_ghost && attr.Ghost) continue;
          if (!attr.Leaf) continue;

          mid.Coord((Iterator<VTKReal>)c);
          VTKReal s = sctl::pow<VTKReal>(0.5, mid.Depth());
          for (Integer j = 0; j < Ncorner; j++) {
            for (Integer i = 0; i < DIM; i++) coord.PushBack(c[i] + ((j & (1u << i)) ? 1 : 0) * s);
            for (Integer i = DIM; i < 3; i++) coord.PushBack(0);
            connect.PushBack(point_cnt);
            connect_cnt++;
            point_cnt++;
          }
          offset.PushBack(connect_cnt);
          if (DIM == 2)
            types.PushBack(8);
          else if (DIM == 3)
            types.PushBack(11);
          else
            types.PushBack(4);
        }
      }
      vtu_data.WriteVTK(fname, comm);
    }

  protected:

    void GetData_(Iterator<Vector<char>>& data, Iterator<Vector<Long>>& cnt, const std::string& name) {
      auto data_ = node_data.find(name);
      const auto cnt_ = node_cnt.find(name);
      SCTL_ASSERT(data_ != node_data.end());
      SCTL_ASSERT( cnt_ != node_cnt .end());
      data = Ptr2Itr<Vector<char>>(&data_->second,1);
      cnt  = Ptr2Itr<Vector<Long>>(& cnt_->second,1);
    }

    static void scan(Vector<Long>& dsp, const Vector<Long>& cnt) {
      dsp.ReInit(cnt.Dim());
      if (cnt.Dim()) dsp[0] = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());
    }

    template <typename A, typename B> struct SortPair {
      int operator<(const SortPair<A, B> &p1) const { return key < p1.key; }
      A key;
      B data;
    };

  private:

    Vector<Morton<DIM>> mins;
    Vector<Morton<DIM>> node_mid;
    Vector<NodeAttr> node_attr;
    Vector<NodeLists> node_lst;

    std::map<std::string, Vector<char>> node_data;
    std::map<std::string, Vector<Long>> node_cnt;

    Vector<Morton<DIM>> user_mid;
    Vector<Long> user_cnt;

    Comm comm;
};

template <class Real, Integer DIM, class BaseTree = Tree<DIM>> class PtTree : public BaseTree {
  public:

    PtTree(const Comm& comm = Comm::Self()) : BaseTree(comm) {}

    ~PtTree() {
      #ifdef SCTL_MEMDEBUG
      for (auto& pair : data_pt_name) {
        Vector<Real> data;
        Vector<Long> cnt;
        this->GetData(data, cnt, pair.second);
        SCTL_ASSERT(scatter_idx.find(pair.second) != scatter_idx.end());
      }
      #endif
    }

    void UpdateRefinement(const Vector<Real>& coord, Long M = 1, bool balance21 = 0, bool periodic = 0) {
      const auto& comm = this->GetComm();
      BaseTree::UpdateRefinement(coord, M, balance21, periodic);

      Long start_node_idx, end_node_idx;
      { // Set start_node_idx, end_node_idx
        const auto& mins = this->GetPartitionMID();
        const auto& node_mid = this->GetNodeMID();
        Integer np = comm.Size();
        Integer rank = comm.Rank();
        start_node_idx = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
        end_node_idx = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
      }

      const auto& mins = this->GetPartitionMID();
      const auto& node_mid = this->GetNodeMID();
      for (const auto& pair : pt_mid) {
        const auto& pt_name = pair.first;
        auto& pt_mid_ = pt_mid[pt_name];
        auto& scatter_idx_ = scatter_idx[pt_name];
        comm.PartitionS(pt_mid_, mins[comm.Rank()]);
        comm.PartitionN(scatter_idx_, pt_mid_.Dim());

        Vector<Long> pt_cnt(node_mid.Dim());
        for (Long i = 0; i < node_mid.Dim(); i++) { // Set pt_cnt
          Long start = std::lower_bound(pt_mid_.begin(), pt_mid_.end(), node_mid[i]) - pt_mid_.begin();
          Long end = std::lower_bound(pt_mid_.begin(), pt_mid_.end(), (i+1==node_mid.Dim() ? Morton<DIM>().Next() : node_mid[i+1])) - pt_mid_.begin();
          if (i == 0) SCTL_ASSERT(start == 0);
          if (i+1 == node_mid.Dim()) SCTL_ASSERT(end == pt_mid_.Dim());
          pt_cnt[i] = end - start;
        }

        for (const auto& pair : data_pt_name) {
          if (pair.second == pt_name) {
            const auto& data_name = pair.first;

            Iterator<Vector<char>> data;
            Iterator<Vector<Long>> cnt;
            this->GetData_(data, cnt, data_name);

            { // Update data
              Long dof = 0;
              { // Set dof
                StaticArray<Long,2> Nl {0, 0}, Ng;
                Nl[0] = data->Dim();
                for (Long i = 0; i < cnt->Dim(); i++) Nl[1] += cnt[0][i];
                comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, Comm::CommOp::SUM);
                dof = Ng[0] / std::max<Long>(Ng[1],1);
              }
              Long offset = 0, count = 0;
              SCTL_ASSERT(0 <= start_node_idx);
              SCTL_ASSERT(start_node_idx <= end_node_idx);
              SCTL_ASSERT(end_node_idx <= cnt->Dim());
              for (Long i = 0; i < start_node_idx; i++) offset += cnt[0][i];
              for (Long i = start_node_idx; i < end_node_idx; i++) count += cnt[0][i];
              offset *= dof;
              count *= dof;

              Vector<char> data_(count, data->begin() + offset);
              comm.PartitionN(data_, pt_mid_.Dim());
              data->Swap(data_);
            }
            cnt[0] = pt_cnt;
          }
        }
      }
    }

    void AddParticles(const std::string& name, const Vector<Real>& coord) {
      const auto& mins = this->GetPartitionMID();
      const auto& node_mid = this->GetNodeMID();
      const auto& comm = this->GetComm();

      SCTL_ASSERT(scatter_idx.find(name) == scatter_idx.end());
      Vector<Long>& scatter_idx_ = scatter_idx[name];

      Long N = coord.Dim() / DIM;
      SCTL_ASSERT(coord.Dim() == N * DIM);
      Nlocal[name] = N;

      Vector<Morton<DIM>>& pt_mid_ = pt_mid[name];
      if (pt_mid_.Dim() != N) pt_mid_.ReInit(N);
      for (Long i = 0; i < N; i++) {
        pt_mid_[i] = Morton<DIM>(coord.begin() + i*DIM);
      }
      comm.SortScatterIndex(pt_mid_, scatter_idx_, &mins[comm.Rank()]);
      comm.ScatterForward(pt_mid_, scatter_idx_);
      AddParticleData(name, name, coord);

      { // Set node_cnt
        Iterator<Vector<char>> data_;
        Iterator<Vector<Long>> cnt_;
        this->GetData_(data_,cnt_,name);
        cnt_[0].ReInit(node_mid.Dim());
        for (Long i = 0; i < node_mid.Dim(); i++) {
          Long start = std::lower_bound(pt_mid_.begin(), pt_mid_.end(), node_mid[i]) - pt_mid_.begin();
          Long end = std::lower_bound(pt_mid_.begin(), pt_mid_.end(), (i+1==node_mid.Dim() ? Morton<DIM>().Next() : node_mid[i+1])) - pt_mid_.begin();
          if (i == 0) SCTL_ASSERT(start == 0);
          if (i+1 == node_mid.Dim()) SCTL_ASSERT(end == pt_mid_.Dim());
          cnt_[0][i] = end - start;
        }
      }
    }

    void AddParticleData(const std::string& data_name, const std::string& particle_name, const Vector<Real>& data) {
      SCTL_ASSERT(scatter_idx.find(particle_name) != scatter_idx.end());
      SCTL_ASSERT(data_pt_name.find(data_name) == data_pt_name.end());
      data_pt_name[data_name] = particle_name;

      Iterator<Vector<char>> data_;
      Iterator<Vector<Long>> cnt_;
      this->AddData(data_name, Vector<Real>(), Vector<Long>());
      this->GetData_(data_,cnt_,data_name);
      { // Set data_[0]
        data_[0].ReInit(data.Dim()*sizeof(Real), (Iterator<char>)data.begin(), true);
        this->GetComm().ScatterForward(data_[0], scatter_idx[particle_name]);
      }
      if (data_name != particle_name) { // Set cnt_[0]
        Vector<Real> pt_coord;
        Vector<Long> pt_cnt;
        this->GetData(pt_coord, pt_cnt, particle_name);
        cnt_[0] = pt_cnt;
      }
    }

    void GetParticleData(Vector<Real>& data, const std::string& data_name) const {
      SCTL_ASSERT(data_pt_name.find(data_name) != data_pt_name.end());
      const std::string& particle_name = data_pt_name.find(data_name)->second;
      SCTL_ASSERT(scatter_idx.find(particle_name) != scatter_idx.end());
      const auto& scatter_idx_ = scatter_idx.find(particle_name)->second;
      const Long Nlocal_ = Nlocal.find(particle_name)->second;

      const auto& mins = this->GetPartitionMID();
      const auto& node_mid = this->GetNodeMID();
      const auto& comm = this->GetComm();

      Long dof;
      Vector<Long> dsp;
      Vector<Long> cnt_;
      Vector<Real> data_;
      this->GetData(data_, cnt_, data_name);
      SCTL_ASSERT(cnt_.Dim() == node_mid.Dim());
      BaseTree::scan(dsp, cnt_);
      { // Set dof
        Long Nn = node_mid.Dim();
        StaticArray<Long,2> Ng, Nl = {data_.Dim(), dsp[Nn-1]+cnt_[Nn-1]};
        comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, Comm::CommOp::SUM);
        dof = Ng[0] / std::max<Long>(Ng[1],1);
      }
      { // Set data
        Integer np = comm.Size();
        Integer rank = comm.Rank();
        Long N0 = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
        Long N1 = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
        Long start = dsp[N0] * dof;
        Long end = (N1<dsp.Dim() ? dsp[N1] : dsp[N1-1]+cnt_[N1-1]) * dof;
        data.ReInit(end-start, data_.begin()+start, true);
        comm.ScatterReverse(data, scatter_idx_, Nlocal_ * dof);
      }
    }

    void DeleteParticleData(const std::string& data_name) {
      SCTL_ASSERT(data_pt_name.find(data_name) != data_pt_name.end());
      auto particle_name = data_pt_name[data_name];
      if (data_name == particle_name) {
        std::vector<std::string> data_name_lst;
        for (auto& pair : data_pt_name) {
          if (pair.second == particle_name) {
            data_name_lst.push_back(pair.first);
          }
        }
        for (auto x : data_name_lst) {
          if (x != particle_name) {
            DeleteParticleData(x);
          }
        }
        Nlocal.erase(particle_name);
      }
      this->DeleteData(data_name);
      data_pt_name.erase(data_name);
    }

    void WriteParticleVTK(std::string fname, std::string data_name, bool show_ghost = false) const {
      typedef typename VTUData::VTKReal VTKReal;
      const auto& node_mid = this->GetNodeMID();
      const auto& node_attr = this->GetNodeAttr();

      VTUData vtu_data;
      if (DIM <= 3) {  // Set vtu data
        SCTL_ASSERT(data_pt_name.find(data_name) != data_pt_name.end());
        std::string particle_name = data_pt_name.find(data_name)->second;

        Vector<Real> pt_coord;
        Vector<Real> pt_value;
        Vector<Long> pt_cnt;
        Vector<Long> pt_dsp;
        Long value_dof = 0;
        { // Set pt_coord, pt_cnt, pt_dsp
          this->GetData(pt_coord, pt_cnt, particle_name);
          Tree<DIM>::scan(pt_dsp, pt_cnt);
        }
        if (particle_name != data_name) { // Set pt_value, value_dof
          Vector<Long> pt_cnt;
          this->GetData(pt_value, pt_cnt, data_name);
          Long Npt = omp_par::reduce(pt_cnt.begin(), pt_cnt.Dim());
          value_dof = pt_value.Dim() / std::max<Long>(Npt,1);
        }

        Vector<VTKReal> &coord = vtu_data.coord;
        Vector<VTKReal> &value = vtu_data.value;

        Vector<int32_t> &connect = vtu_data.connect;
        Vector<int32_t> &offset = vtu_data.offset;
        Vector<uint8_t> &types = vtu_data.types;

        Long point_cnt = coord.Dim() / DIM;
        Long connect_cnt = connect.Dim();
        value.ReInit(point_cnt * value_dof);
        value.SetZero();

        SCTL_ASSERT(node_mid.Dim() == node_attr.Dim());
        SCTL_ASSERT(node_mid.Dim() == pt_cnt.Dim());
        for (Long i = 0; i < node_mid.Dim(); i++) {
          if (!show_ghost && node_attr[i].Ghost) continue;
          if (!node_attr[i].Leaf) continue;

          for (Long j = 0; j < pt_cnt[i]; j++) {
            ConstIterator<Real> pt_coord_ = pt_coord.begin() + (pt_dsp[i] + j) * DIM;
            ConstIterator<Real> pt_value_ = (value_dof ? pt_value.begin() + (pt_dsp[i] + j) * value_dof : NullIterator<Real>());

            for (Integer k = 0; k < DIM; k++) coord.PushBack((VTKReal)pt_coord_[k]);
            for (Integer k = DIM; k < 3; k++) coord.PushBack(0);
            for (Integer k = 0; k < value_dof; k++) value.PushBack((VTKReal)pt_value_[k]);
            connect.PushBack(point_cnt);
            connect_cnt++;
            point_cnt++;

            offset.PushBack(connect_cnt);
            types.PushBack(1);
          }
        }
      }
      vtu_data.WriteVTK(fname, this->GetComm());
    }

  private:

    std::map<std::string, Long> Nlocal;
    std::map<std::string, Vector<Morton<DIM>>> pt_mid;
    std::map<std::string, Vector<Long>> scatter_idx;
    std::map<std::string, std::string> data_pt_name;
};

}

#endif //_SCTL_TREE_
