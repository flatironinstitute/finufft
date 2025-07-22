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
#include <intrin.h>
static bool cpuid_subleaf(uint32_t leaf, uint32_t sub, uint32_t &eax, uint32_t &ebx,
                          uint32_t &ecx, uint32_t &edx) {
  int r[4];
  __cpuidex(r, static_cast<int>(leaf), static_cast<int>(sub));
  eax = static_cast<uint32_t>(r[0]);
  ebx = static_cast<uint32_t>(r[1]);
  ecx = static_cast<uint32_t>(r[2]);
  edx = static_cast<uint32_t>(r[3]);
  return true;
}
struct AffinityGuard {
  HANDLE th      = GetCurrentThread();
  DWORD_PTR orig = 0;
  explicit AffinityGuard(unsigned cpu) {
    orig = SetThreadAffinityMask(th, DWORD_PTR(1) << cpu);
    if (!orig) orig = 1;
  }
  ~AffinityGuard() { SetThreadAffinityMask(th, orig); }
};
static unsigned count_physical_cores_win() {
  uint32_t a = 0, b = 0, c = 0, d = 0;
  cpuid_subleaf(1, 0, a, b, c, d);
  const unsigned logical = (b >> 16) & 0xFF;
  std::set<unsigned> core_ids;
  for (unsigned i = 0; i < logical; ++i) {
    AffinityGuard pin(i);
    cpuid_subleaf(1, 0, a, b, c, d);
    const unsigned apic = (b >> 24) & 0xFF;
    uint32_t shift      = 0;
    for (uint32_t lvl = 0;; ++lvl) {
      if (!cpuid_subleaf(0x0B, lvl, a, b, c, d)) break;
      const uint32_t typ = (c >> 8) & 0xFF;
      if (typ == 1) {
        shift = a & 0x1F;
        break;
      }
      if (typ == 0) break;
    }
    core_ids.insert(shift ? (apic >> shift) : apic);
  }
  return static_cast<unsigned>(core_ids.size());
}
#else
static unsigned count_physical_cores_win() {
  // Fallback: use logical processor count as a proxy
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
}
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
bool cpuid_subleaf(uint32_t leaf, uint32_t sub, uint32_t &eax, uint32_t &ebx,
                   uint32_t &ecx, uint32_t &edx) {
  return __get_cpuid_count(leaf, sub, &eax, &ebx, &ecx, &edx);
}
void pin_cpu(unsigned idx) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(idx, &s);
  pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}
unsigned count_physical_cores_linux() {
  uint32_t a = 0, b = 0, c = 0, d = 0;
  if (!cpuid_subleaf(1, 0, a, b, c, d)) return 0;
  const unsigned logical = (b >> 16) & 0xFF;
  cpu_set_t orig;
  pthread_getaffinity_np(pthread_self(), sizeof(orig), &orig);
  std::set<unsigned> core_ids;
  for (unsigned i = 0; i < logical; ++i) {
    pin_cpu(i);
    cpuid_subleaf(1, 0, a, b, c, d);
    const unsigned apic = (b >> 24) & 0xFF;
    uint32_t shift      = 0;
    for (uint32_t lvl = 0;; ++lvl) {
      if (!cpuid_subleaf(0x0B, lvl, a, b, c, d)) break;
      const uint32_t typ = (c >> 8) & 0xFF;
      if (typ == 1) {
        shift = a & 0x1F;
        break;
      }
      if (typ == 0) break;
    }
    core_ids.insert(shift ? (apic >> shift) : apic);
  }
  pthread_setaffinity_np(pthread_self(), sizeof(orig), &orig);
  return static_cast<unsigned>(core_ids.size());
}
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
  if (debug > 1) std::cout << "[finufft::utils] physical cores=" << n << "\n";
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
  if (debug > 1) std::cout << "[finufft::utils] allowed cores=" << n << "\n";
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
  const auto physical = getPhysicalCoreCount(debug);
  const auto allowed  = getAllowedCoreCount(debug);
  const auto optimal  = std::min(physical, allowed);
  return optimal ? optimal : MY_OMP_GET_MAX_THREADS();
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
