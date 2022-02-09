#ifndef _SCTL_PROFILE_HPP_
#define _SCTL_PROFILE_HPP_

#include <sctl/common.hpp>

#include <string>
#include <vector>
#include <stack>

#ifndef SCTL_PROFILE
#define SCTL_PROFILE -1
#endif

namespace SCTL_NAMESPACE {

class Comm;

class Profile {
 public:
  static Long Add_MEM(Long inc);

  static Long Add_FLOP(Long inc);

  static bool Enable(bool state);

  static void Tic(const char* name_, const Comm* comm_ = nullptr, bool sync_ = false, Integer level = 0);

  static void Toc();

  static void print(const Comm* comm_ = nullptr);

  static void reset();

 private:
  struct ProfileData {
    Long MEM;
    Long FLOP;
    bool enable_state;
    std::stack<bool> sync;
    std::stack<std::string> name;
    std::stack<const Comm*> comm;
    std::vector<Long> max_mem;

    Integer enable_depth;
    std::stack<int> verb_level;

    std::vector<bool> e_log;
    std::vector<bool> s_log;
    std::vector<std::string> n_log;
    std::vector<double> t_log;
    std::vector<Long> f_log;
    std::vector<Long> m_log;
    std::vector<Long> max_m_log;

    ProfileData() : MEM(0), FLOP(0), enable_state(false), enable_depth(0) {}
  };

  static inline ProfileData& ProfData() {
    static ProfileData p;
    return p;
  }
};

}  // end namespace

#include SCTL_INCLUDE(profile.txx)

#endif  //_SCTL_PROFILE_HPP_
