#include SCTL_INCLUDE(comm.hpp)

#include <omp.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstdlib>

namespace SCTL_NAMESPACE {

#if SCTL_PROFILE >= 0

inline Long Profile::Add_MEM(Long inc) {
  std::vector<Long>& max_mem = ProfData().max_mem;
  Long& MEM = ProfData().MEM;
  Long orig_val = MEM;
#pragma omp atomic update
  MEM += inc;
  for (Integer i = max_mem.size() - 1; i >= 0 && max_mem[i] < MEM; i--) max_mem[i] = MEM;
  return orig_val;
}

inline Long Profile::Add_FLOP(Long inc) {
  Long& FLOP = ProfData().FLOP;
  Long orig_val = FLOP;
#pragma omp atomic update
  FLOP += inc;
  return orig_val;
}

inline bool Profile::Enable(bool state) {
  bool& enable_state = ProfData().enable_state;
  bool orig_val = enable_state;
  enable_state = state;
  return orig_val;
}

inline void Profile::Tic(const char* name_, const Comm* comm_, bool sync_, Integer verbose) {
  ProfileData& prof = ProfData();
  if (!prof.enable_state) return;
  // sync_=true;
  if (verbose <= SCTL_PROFILE && (Integer)prof.verb_level.size() == prof.enable_depth) {
    if (comm_ != nullptr && sync_) comm_->Barrier();
#ifdef SCTL_VERBOSE
    Integer rank = 0;
    if (comm_ != nullptr) rank = comm_->Rank();
    if (!rank) {
      for (size_t i = 0; i < prof.name.size(); i++) std::cout << "    ";
      std::cout << "\033[1;31m" << name_ << "\033[0m {\n";
    }
#endif
    prof.name.push(name_);
    prof.comm.push(comm_);
    prof.sync.push(sync_);
    prof.max_mem.push_back(prof.MEM);

    prof.e_log.push_back(true);
    prof.s_log.push_back(sync_);
    prof.n_log.push_back(prof.name.top());
    prof.t_log.push_back(omp_get_wtime());
    prof.f_log.push_back(prof.FLOP);

    prof.m_log.push_back(prof.MEM);
    prof.max_m_log.push_back(prof.MEM);
    prof.enable_depth++;
  }
  prof.verb_level.push(verbose);
}

inline void Profile::Toc() {
  ProfileData& prof = ProfData();
  if (!prof.enable_state) return;
  SCTL_ASSERT_MSG(!prof.verb_level.empty(), "Unbalanced extra Toc()");
  if (prof.verb_level.top() <= SCTL_PROFILE && (Integer)prof.verb_level.size() == prof.enable_depth) {
    SCTL_ASSERT_MSG(!prof.name.empty() && !prof.comm.empty() && !prof.sync.empty() && !prof.max_mem.empty(), "Unbalanced extra Toc()");
    std::string name_ = prof.name.top();
    const Comm* comm_ = prof.comm.top();
    bool sync_ = prof.sync.top();
    // sync_=true;

    prof.e_log.push_back(false);
    prof.s_log.push_back(sync_);
    prof.n_log.push_back(name_);
    prof.t_log.push_back(omp_get_wtime());
    prof.f_log.push_back(prof.FLOP);

    prof.m_log.push_back(prof.MEM);
    prof.max_m_log.push_back(prof.max_mem.back());

#ifndef NDEBUG
    if (comm_ != nullptr && sync_) comm_->Barrier();
#endif
    prof.name.pop();
    prof.comm.pop();
    prof.sync.pop();
    prof.max_mem.pop_back();

#ifdef SCTL_VERBOSE
    Integer rank = 0;
    if (comm_ != nullptr) rank = comm_->Rank();
    if (!rank) {
      for (size_t i = 0; i < prof.name.size(); i++) std::cout << "    ";
      std::cout << "}\n";
    }
#endif
    prof.enable_depth--;
  }
  prof.verb_level.pop();
}

inline void Profile::print(const Comm* comm_) {
  ProfileData& prof = ProfData();
  SCTL_ASSERT_MSG(prof.name.empty(), "Missing balancing Toc()");

  Comm c_self = Comm::Self();
  if (comm_ == nullptr) comm_ = &c_self;
  comm_->Barrier();

  Integer np, rank;
  np = comm_->Size();
  rank = comm_->Rank();

  std::stack<double> tt;
  std::stack<Long> ff;
  std::stack<Long> mm;
  Integer width = 10;
  size_t level = 0;
  if (!rank && prof.e_log.size() > 0) {
    std::cout << "\n" << std::setw(width * 3 - 2 * level) << " ";
    if (np == 1) {
      std::cout << "  " << std::setw(width) << "t";
      std::cout << "  " << std::setw(width) << "f";
      std::cout << "  " << std::setw(width) << "f/s";
    } else {
      std::cout << "  " << std::setw(width) << "t_min";
      std::cout << "  " << std::setw(width) << "t_avg";
      std::cout << "  " << std::setw(width) << "t_max";

      std::cout << "  " << std::setw(width) << "f_min";
      std::cout << "  " << std::setw(width) << "f_avg";
      std::cout << "  " << std::setw(width) << "f_max";

      std::cout << "  " << std::setw(width) << "f/s_min";
      std::cout << "  " << std::setw(width) << "f/s_max";
      std::cout << "  " << std::setw(width) << "f/s_total";
    }

    std::cout << "  " << std::setw(width) << "m_init";
    std::cout << "  " << std::setw(width) << "m_max";
    std::cout << "  " << std::setw(width) << "m_final" << '\n';
  }

  std::stack<std::string> out_stack;
  std::string s;
  out_stack.push(s);
  for (size_t i = 0; i < prof.e_log.size(); i++) {
    if (prof.e_log[i]) {
      level++;
      tt.push(prof.t_log[i]);
      ff.push(prof.f_log[i]);
      mm.push(prof.m_log[i]);

      std::string ss;
      out_stack.push(ss);
    } else {
      double t0 = prof.t_log[i] - tt.top();
      tt.pop();
      double f0 = (double)(prof.f_log[i] - ff.top()) * 1e-9;
      ff.pop();
      double fs0 = f0 / t0;
      double t_max, t_min, t_sum, t_avg;
      double f_max, f_min, f_sum, f_avg;
      double fs_max, fs_min, fs_sum;  //, fs_avg;
      double m_init, m_max, m_final;
      comm_->Allreduce(Ptr2ConstItr<double>(&t0, 1), Ptr2Itr<double>(&t_max, 1), 1, Comm::CommOp::MAX);
      comm_->Allreduce(Ptr2ConstItr<double>(&f0, 1), Ptr2Itr<double>(&f_max, 1), 1, Comm::CommOp::MAX);
      comm_->Allreduce(Ptr2ConstItr<double>(&fs0, 1), Ptr2Itr<double>(&fs_max, 1), 1, Comm::CommOp::MAX);

      comm_->Allreduce(Ptr2ConstItr<double>(&t0, 1), Ptr2Itr<double>(&t_min, 1), 1, Comm::CommOp::MIN);
      comm_->Allreduce(Ptr2ConstItr<double>(&f0, 1), Ptr2Itr<double>(&f_min, 1), 1, Comm::CommOp::MIN);
      comm_->Allreduce(Ptr2ConstItr<double>(&fs0, 1), Ptr2Itr<double>(&fs_min, 1), 1, Comm::CommOp::MIN);

      comm_->Allreduce(Ptr2ConstItr<double>(&t0, 1), Ptr2Itr<double>(&t_sum, 1), 1, Comm::CommOp::SUM);
      comm_->Allreduce(Ptr2ConstItr<double>(&f0, 1), Ptr2Itr<double>(&f_sum, 1), 1, Comm::CommOp::SUM);

      m_final = (double)prof.m_log[i] * 1e-9;
      m_init = (double)mm.top() * 1e-9;
      mm.pop();
      m_max = (double)prof.max_m_log[i] * 1e-9;

      t_avg = t_sum / np;
      f_avg = f_sum / np;
      // fs_avg=f_avg/t_max;
      fs_sum = f_sum / t_max;

      if (!rank) {
        std::string s0 = out_stack.top();
        out_stack.pop();
        std::string s1 = out_stack.top();
        out_stack.pop();
        std::stringstream ss(std::stringstream::in | std::stringstream::out);
        ss << setiosflags(std::ios::fixed) << std::setprecision(4) << std::setiosflags(std::ios::left);

        for (size_t j = 0; j < level - 1; j++) {
          size_t l = i + 1;
          size_t k = level - 1;
          while (k > j && l < prof.e_log.size()) {
            k += (prof.e_log[l] ? 1 : -1);
            l++;
          }
          if (l < prof.e_log.size() ? prof.e_log[l] : false)
            ss << "| ";
          else
            ss << "  ";
        }
        ss << "+-";
        ss << std::setw(width * 3 - 2 * level) << prof.n_log[i];
        ss << std::setiosflags(std::ios::right);
        if (np == 1) {
          ss << "  " << std::setw(width) << t_avg;
          ss << "  " << std::setw(width) << f_avg;
          ss << "  " << std::setw(width) << fs_sum;
        } else {
          ss << "  " << std::setw(width) << t_min;
          ss << "  " << std::setw(width) << t_avg;
          ss << "  " << std::setw(width) << t_max;

          ss << "  " << std::setw(width) << f_min;
          ss << "  " << std::setw(width) << f_avg;
          ss << "  " << std::setw(width) << f_max;

          ss << "  " << std::setw(width) << fs_min;
          // ss<<"  "<<std::setw(width)<<fs_avg;
          ss << "  " << std::setw(width) << fs_max;
          ss << "  " << std::setw(width) << fs_sum;
        }

        ss << "  " << std::setw(width) << m_init;
        ss << "  " << std::setw(width) << m_max;
        ss << "  " << std::setw(width) << m_final << '\n';

        s1 += ss.str() + s0;
        if (!s0.empty() && (i + 1 < prof.e_log.size() ? prof.e_log[i + 1] : false)) {
          for (size_t j = 0; j < level; j++) {
            size_t l = i + 1;
            size_t k = level - 1;
            while (k > j && l < prof.e_log.size()) {
              k += (prof.e_log[l] ? 1 : -1);
              l++;
            }
            if (l < prof.e_log.size() ? prof.e_log[l] : false)
              s1 += "| ";
            else
              s1 += "  ";
          }
          s1 += "\n";
        }  // */
        out_stack.push(s1);
      }
      level--;
    }
  }
  if (!rank) std::cout << out_stack.top() << '\n';

  reset();
}

inline void Profile::reset() {
  ProfileData& prof = ProfData();
  prof.FLOP = 0;
  while (!prof.sync.empty()) prof.sync.pop();
  while (!prof.name.empty()) prof.name.pop();
  while (!prof.comm.empty()) prof.comm.pop();

  prof.e_log.clear();
  prof.s_log.clear();
  prof.n_log.clear();
  prof.t_log.clear();
  prof.f_log.clear();
  prof.m_log.clear();
  prof.max_m_log.clear();
}

#else

inline Long Profile::Add_FLOP(Long inc) { return 0; }

inline Long Profile::Add_MEM(Long inc) { return 0; }

inline bool Profile::Enable(bool state) { return false; }

inline void Profile::Tic(const char* name_, const Comm* comm_, bool sync_, Integer verbose) { }

inline void Profile::Toc() { }

inline void Profile::print(const Comm* comm_) { }

inline void Profile::reset() { }

#endif

}  // end namespace
