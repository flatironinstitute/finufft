// Low-level array manipulations, quadrature, timer, and OMP helpers,
// that are precision-independent (no FLT allowed in argument lists).

#include "finufft/finufft_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace finufft::utils {

// ------------------- Math Utilities -------------------

BIGINT next235even(BIGINT n) {
  if (n <= 2) return 2;
  if (n % 2 == 1) n += 1;
  BIGINT nplus = n - 2, numdiv = 2;
  while (numdiv > 1) {
    nplus += 2;
    numdiv = nplus;
    for (int f : {2, 3, 5})
      while (numdiv % f == 0) numdiv /= f;
  }
  return nplus;
}

std::tuple<double, double> leg_eval(int n, double x) {
  if (n == 0) return {1.0, 0.0};
  if (n == 1) return {x, 1.0};
  double p0 = 0.0, p1 = 1.0, p2 = x;
  for (int i = 1; i < n; i++) {
    p0 = p1;
    p1 = p2;
    p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1);
  }
  return {p2, n * (x * p2 - p1) / (x * x - 1)};
}

void gaussquad(int n, double *xgl, double *wgl) {
  double x = 0, dx = 0;
  int convcount = 0;
  xgl[n / 2]    = 0;
  for (int i = 0; i < n / 2; i++) {
    convcount = 0;
    x         = cos((2 * i + 1) * PI / (2 * n));
    while (true) {
      auto [p, dp] = leg_eval(n, x);
      dx           = -p / dp;
      x += dx;
      if (std::abs(dx) < 1e-14) convcount++;
      if (convcount == 3) break;
    }
    xgl[i]         = -x;
    xgl[n - i - 1] = x;
  }
  for (int i = 0; i < n / 2 + 1; i++) {
    auto [_, dp]   = leg_eval(n, xgl[i]);
    auto [p, __]   = leg_eval(n + 1, xgl[i]);
    wgl[i]         = -2 / ((n + 1) * dp * p);
    wgl[n - i - 1] = wgl[i];
  }
}

// ------------------- Timer Utilities -------------------

void CNTime::start() {
  initial = double(std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count()) *
            1e-6;
}

double CNTime::restart() {
  double delta = elapsedsec();
  start();
  return delta;
}

double CNTime::elapsedsec() const {
  std::uint64_t now = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  return double(now) * 1e-6 - initial;
}

// ------------------- Platform-Specific Thread/Core Utilities -------------------

#ifdef _OPENMP

// --- Windows ---
#ifdef _WIN32
#include <windows.h>
#if defined(__i386__) || defined(__x86_64__)
static unsigned count_physical_cores_win() {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);

  std::vector<uint8_t> buffer(length);
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info =
      reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data());

  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
    std::cerr << "Failed to query processor information.\n";
    return 0;
  }

  size_t offset          = 0;
  int physical_cores     = 0;
  int logical_processors = 0;

  while (offset < length) {
    auto *coreInfo = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
        buffer.data() + offset);
    if (coreInfo->Relationship == RelationProcessorCore) {
      ++physical_cores;
      // Count the number of logical processors (bits set in the mask)
      GROUP_AFFINITY &affinity = coreInfo->Processor.GroupMask[0];
      DWORD64 mask             = affinity.Mask;
      int lp_count             = 0;
      while (mask) {
        lp_count += mask & 1;
        mask >>= 1;
      }
      logical_processors += lp_count;
    }
    offset += coreInfo->Size;
  }
  return physical_cores;
}
#else
// on ARM there is no hyperthreading so we can use the logical count directly
static unsigned count_physical_cores_win() { MY_OMP_GET_MAX_THREADS(); }
#endif
static unsigned count_allowed_cores_win() {
  DWORD_PTR processMask = 0, systemMask = 0;
  if (GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
    unsigned cnt = 0;
    for (; processMask; processMask &= (processMask - 1)) ++cnt;
    return cnt;
  }
  return 0;
}
#endif

// --- macOS ---
#ifdef __APPLE__
#include <sys/sysctl.h>
static unsigned count_physical_cores_mac() {
  int n     = 0;
  size_t sz = sizeof(n);
  if (!sysctlbyname("hw.physicalcpu", &n, &sz, nullptr, 0) && n > 0) return n;
  return 0;
}
#endif

// --- Linux ---
#ifdef __linux__
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
static bool cpuid_subleaf(uint32_t leaf, uint32_t subleaf, uint32_t &eax, uint32_t &ebx,
                          uint32_t &ecx, uint32_t &edx) {
  // __get_cpuid_max returns max basic leaf
  uint32_t max_leaf = __get_cpuid_max(0, nullptr);
  if (leaf > max_leaf) return false;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  return true;
}
static bool pin_cpu(unsigned idx) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(idx, &s);
  return pthread_setaffinity_np(pthread_self(), sizeof(s), &s) == 0;
}

enum class TopoMethod { Leaf1F, Leaf0B, Unsupported };

static TopoMethod detect_topology_method() {
  uint32_t a, b, c, d;
  if (!cpuid_subleaf(0, 0, a, b, c, d)) return TopoMethod::Unsupported;
  uint32_t max_leaf = a;

  if (max_leaf >= 0x1F) {
    cpuid_subleaf(0x1F, 0, a, b, c, d);
    if (((c >> 8) & 0xFF) != 0) return TopoMethod::Leaf1F;
  }
  if (max_leaf >= 0x0B) {
    cpuid_subleaf(0x0B, 0, a, b, c, d);
    if (((c >> 8) & 0xFF) != 0) return TopoMethod::Leaf0B;
  }
  return TopoMethod::Unsupported;
}

static unsigned count_physical_cores_cpuid_linux() {
  uint32_t a, b, c, d;
  if (!cpuid_subleaf(1, 0, a, b, c, d)) return 0;
  unsigned logical = (b >> 16) & 0xFF;

  cpu_set_t orig;
  pthread_getaffinity_np(pthread_self(), sizeof(orig), &orig);
  std::set<unsigned> cores;
  TopoMethod method = detect_topology_method();
  if (method == TopoMethod::Unsupported) return 0;

  for (unsigned i = 0; i < logical; ++i) {
    if (!pin_cpu(i)) continue;
    cpuid_subleaf(1, 0, a, b, c, d);
    unsigned apic = (b >> 24) & 0xFF;

    uint32_t shift = 0;
    for (uint32_t lvl = 0;; ++lvl) {
      uint32_t leaf = (method == TopoMethod::Leaf1F ? 0x1F : 0x0B);
      if (!cpuid_subleaf(leaf, lvl, a, b, c, d)) break;
      uint32_t type = (c >> 8) & 0xFF;
      if (type == 1) {
        shift = a & 0x1F;
        break;
      }
      if (type == 0) break;
    }
    cores.insert(shift ? (apic >> shift) : apic);
  }
  pthread_setaffinity_np(pthread_self(), sizeof(orig), &orig);
  return cores.size();
}

static unsigned count_physical_cores_linux() {
#if defined(__i386__) && defined(__GNUC__)
  uint32_t eflags;
  __asm__ volatile("pushfl\n\t"
                   "popl %%eax\n\t"
                   "movl %%eax, %%ecx\n\t"
                   "xorl $0x200000, %%eax\n\t"
                   "pushl %%eax\n\t"
                   "popfl\n\t"
                   "pushfl\n\t"
                   "popl %%eax\n\t"
                   "xorl %%ecx, %%eax\n\t"
                   : "=a"(eflags)
                   :
                   : "ecx");
  const auto has_cpuid = (eflags & 0x200000);
#else
  uint64_t eflags;
  __asm__ volatile("pushfq\n\t"
                   "pop %%rax\n\t"
                   "mov %%rax, %%rcx\n\t"
                   "xor $0x200000, %%rax\n\t"
                   "push %%rax\n\t"
                   "popfq\n\t"
                   "pushfq\n\t"
                   "pop %%rax\n\t"
                   "xor %%rcx, %%rax\n\t"
                   : "=a"(eflags)
                   :
                   : "rcx");
  const auto has_cpuid = (eflags & 0x200000);
#endif
  if (has_cpuid) {
    return count_physical_cores_cpuid_linux();
  }
  return MY_OMP_GET_MAX_THREADS(); // fallback to OpenMP max threads
}
#else
// on ARM there is no hyperthreading so we can use the logical count directly
static unsigned count_physical_cores_linux() { return MY_OMP_GET_MAX_THREADS(); }
#endif
static unsigned count_allowed_cores_linux() {
  cpu_set_t cpus;
  if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0) return CPU_COUNT(&cpus);
  return 0;
}
#endif

static unsigned getPhysicalCoreCount(int debug) {
#ifdef _WIN32
  unsigned n = count_physical_cores_win();
#elif defined(__APPLE__)
  unsigned n = count_physical_cores_mac();
#elif defined(__linux__)
  unsigned n = count_physical_cores_linux();
#else
  unsigned n = MY_OMP_GET_MAX_THREADS();
#endif
  if (debug > 1) std::cout << "[FINUFFT_PLAN_T] physical cores=" << n << "\n";
  return n;
}

static unsigned getAllowedCoreCount(int debug) {
#ifdef _WIN32
  unsigned n = count_allowed_cores_win();
#elif defined(__APPLE__)
  unsigned n = MY_OMP_GET_MAX_THREADS();
#elif defined(__linux__)
  unsigned n = count_allowed_cores_linux();
#else
  unsigned n = MY_OMP_GET_MAX_THREADS();
#endif
  if (debug > 1) std::cout << "[FINUFFT_PLAN_T] allowed cores=" << n << "\n";
  return n;
}

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

unsigned getOptimalThreadCount(int debug) {
  if (const auto v = std::getenv("OMP_NUM_THREADS")) {
    try {
      return std::stoi(v);
    } catch (...) {
      std::cerr << "[FINUFFT_PLAN_T] OMP_NUM_THREADS env var is not a valid integer: "
                << v << "\n";
      std::cerr << "[FINUFFT_PLAN_T] Using default thread count instead.\n";
    }
  }
  const auto max_threads = static_cast<unsigned>(MY_OMP_GET_MAX_THREADS());
  const auto safe_call   = [max_threads](auto F, auto... args) -> unsigned {
    try {
      const auto threads = F(args...);
      return std::min(threads, max_threads);
    } catch (...) {
      std::cerr << "[FINUFFT_PLAN_T] Error while determining thread count: "
                << "Falling back to OpenMP.\n";
      return MY_OMP_GET_MAX_THREADS();
    }
  };
  const auto physical = safe_call(getPhysicalCoreCount, debug);
  const auto allowed  = safe_call(getAllowedCoreCount, debug);
  const auto optimal  = std::min(physical, allowed);
  return optimal ? optimal : max_threads;
}

#endif // _OPENMP

#ifdef _WIN32
int rand_r(unsigned *seedp) {
  std::mt19937_64 gen(*seedp);
  *seedp = gen();
  return int(gen() & 0x7FFFFFFF);
}
#endif

} // namespace finufft::utils
