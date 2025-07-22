// Low-level array manipulations, quadrature, timer, and OMP helpers,
// that are precision-independent (no FLT allowed in argument lists).

// For self-test see ../test/testutils.cpp

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

#ifdef _OPENMP
#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#elif defined(__linux__)
#include <cpuid.h>
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <cpuid.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <cpuid.h>
#endif
#endif

namespace finufft::utils {
BIGINT next235even(BIGINT n)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth"). Adapted from fortran in hellskitchen.  Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
{
  if (n <= 2) return 2;
  if (n % 2 == 1) n += 1;                // even
  BIGINT nplus  = n - 2;                 // to cancel out the +=2 at start of loop
  BIGINT numdiv = 2;                     // a dummy that is >1
  while (numdiv > 1) {
    nplus += 2;                          // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0) numdiv /= 2; // remove all factors of 2,3,5...
    while (numdiv % 3 == 0) numdiv /= 3;
    while (numdiv % 5 == 0) numdiv /= 5;
  }
  return nplus;
}

void gaussquad(int n, double *xgl, double *wgl) {
  // n-node Gauss-Legendre quadrature, adapted from a code by Jason Kaye (2022-2023),
  // from the utils file of https://github.com/flatironinstitute/cppdlr version 1.2,
  // which is Apache-2 licensed. It uses Newton iteration from Chebyshev points.
  // Double-precision only.
  // Adapted by Barnett 6/8/25 to write nodes (xgl) and weights (wgl) into arrays
  // that the user must pre-allocate to length at least n.

  double x = 0, dx = 0;
  int convcount = 0;

  // Get Gauss-Legendre nodes
  xgl[n / 2] = 0;                   // If odd number of nodes, middle node is 0
  for (int i = 0; i < n / 2; i++) { // Loop through nodes
    convcount = 0;
    x         = cos((2 * i + 1) * PI / (2 * n)); // Initial guess: Chebyshev node
    while (true) {                               // Newton iteration
      auto [p, dp] = leg_eval(n, x);
      dx           = -p / dp;
      x += dx; // Newton step
      if (std::abs(dx) < 1e-14) {
        convcount++;
      }
      if (convcount == 3) {
        break;
      } // If convergence tol hit 3 times, stop
    }
    xgl[i]         = -x;
    xgl[n - i - 1] = x; // Symmetric nodes
  }

  // Get Gauss-Legendre weights from formula
  // w_i = -2 / ((n+1)*P_n'(x_i)*P_{n+1}(x_i)) (Atkinson '89, pg. 276)
  for (int i = 0; i < n / 2 + 1; i++) {
    auto [junk1, dp] = leg_eval(n, xgl[i]);
    auto [p, junk2]  = leg_eval(n + 1, xgl[i]); // This is a bit inefficient, but who
    // cares...
    wgl[i]         = -2 / ((n + 1) * dp * p);
    wgl[n - i - 1] = wgl[i];
  }
}

std::tuple<double, double> leg_eval(int n, double x) {
  // return Legendre polynomial P_n(x) and its derivative P'_n(x).
  // Uses Legendre three-term recurrence.
  // Used by gaussquad above, with which it shares the same authorship and source.

  if (n == 0) {
    return {1.0, 0.0};
  }
  if (n == 1) {
    return {x, 1.0};
  }
  // Three-term recurrence and formula for derivative
  double p0 = 0.0, p1 = 1.0, p2 = x;
  for (int i = 1; i < n; i++) {
    p0 = p1;
    p1 = p2;
    p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1);
  }
  return {p2, n * (x * p2 - p1) / (x * x - 1)};
}

// ----------------------- helpers for timing (always stay double prec) ------

void CNTime::start() {
  initial = double(std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count()) *
            1e-6;
}

double CNTime::restart()
// Barnett changed to returning in sec
{
  double delta = elapsedsec();
  start();
  return delta;
}

double CNTime::elapsedsec() const
// returns answers as double, in seconds, to microsec accuracy. Barnett 5/22/18
{
  std::uint64_t now = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  const double nowsec = double(now) * 1e-6;
  return nowsec - initial;
}
#ifdef _OPENMP
namespace {

bool cpuid_subleaf(uint32_t leaf,
                   uint32_t sub,
                   uint32_t &eax,
                   uint32_t &ebx,
                   uint32_t &ecx,
                   uint32_t &edx) {
#ifdef _WIN32
  int r[4];
  __cpuidex(r, int(leaf), int(sub));
  eax = r[0];
  ebx = r[1];
  ecx = r[2];
  edx = r[3];
  return true;
#else
  return __get_cpuid_count(leaf, sub, &eax, &ebx, &ecx, &edx);
#endif
}

#ifdef _WIN32
unsigned physical_cores_windows() {
  uint32_t a, b, c, d;
  cpuid_subleaf(1, 0, a, b, c, d);
  unsigned logical = (b >> 16) & 0xFF;
  HANDLE th        = GetCurrentThread();
  DWORD_PTR orig   = SetThreadAffinityMask(th, ~DWORD_PTR(0));
  if (!orig) orig = 1;
  std::set<unsigned> cores;
  for (unsigned i = 0; i < logical; ++i) {
    SetThreadAffinityMask(th, DWORD_PTR(1) << i);
    cpuid_subleaf(1, 0, a, b, c, d);
    unsigned apic  = (b >> 24) & 0xFF;
    uint32_t shift = 0;
    for (uint32_t lvl = 0;; ++lvl) {
      if (!cpuid_subleaf(0x0B, lvl, a, b, c, d)) break;
      uint32_t t = (c >> 8) & 0xFF;
      if (t == 1) {
        shift = a & 0x1F;
        break;
      }
      if (t == 0) break;
    }
    cores.insert(shift ? (apic >> shift) : apic);
  }
  SetThreadAffinityMask(th, orig);
  return unsigned(cores.size());
}
#endif

#if !defined(_WIN32)
void pin_cpu(unsigned idx) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(idx, &s);
  pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}
unsigned physical_cores_posix() {
  uint32_t a, b, c, d;
  cpuid_subleaf(1, 0, a, b, c, d);
  unsigned logical = (b >> 16) & 0xFF;
  cpu_set_t orig;
  pthread_getaffinity_np(pthread_self(), sizeof(orig), &orig);
  std::set<unsigned> cores;
  for (unsigned i = 0; i < logical; ++i) {
    pin_cpu(i);
    cpuid_subleaf(1, 0, a, b, c, d);
    unsigned apic  = (b >> 24) & 0xFF;
    uint32_t shift = 0;
    for (uint32_t lvl = 0;; ++lvl) {
      if (!cpuid_subleaf(0x0B, lvl, a, b, c, d)) break;
      uint32_t t = (c >> 8) & 0xFF;
      if (t == 1) {
        shift = a & 0x1F;
        break;
      }
      if (t == 0) break;
    }
    cores.insert(shift ? (apic >> shift) : apic);
  }
  pthread_setaffinity_np(pthread_self(), sizeof(orig), &orig);
  return unsigned(cores.size());
}
#endif

#ifdef __APPLE__
unsigned physical_cores_sysctl() {
  int cores = 0;
  size_t sz = sizeof(cores);
  if (!sysctlbyname("hw.physicalcpu", &cores, &sz, nullptr, 0) && cores > 0)
    return unsigned(cores);
  return 0;
}
#endif

#ifdef __linux__
unsigned physical_cores_sysfs() {
  const char *base = "/sys/devices/system/cpu";
  DIR *dir         = opendir(base);
  if (!dir) {
    // Could not open directory
    return 0;
  }

  std::set<std::pair<int, int>> uniq;
  dirent *entry{nullptr};
  while ((entry = readdir(dir)) != nullptr) {
    // Skip "." and ".."
    if (entry->d_name[0] == '.') continue;

    // Only consider entries that look like "cpuN"
    std::string name(entry->d_name);
    if (name.size() < 4 || name.compare(0, 3, "cpu") != 0) continue;

    // Build full path to the CPU directory
    std::string cpu_path = std::string(base) + "/" + name;

    // Ensure it's actually a directory
    struct stat st{};
    if (stat(cpu_path.c_str(), &st) < 0 || !S_ISDIR(st.st_mode)) continue;

    // Read core_id and cluster_id
    int core_id = -1, cluster_id = -1;
    {
      std::ifstream ifs(cpu_path + "/topology/core_id");
      if (ifs) ifs >> core_id;
    }
    {
      std::ifstream ifs(cpu_path + "/topology/cluster_id");
      if (ifs) ifs >> cluster_id;
    }

    if (core_id >= 0 && cluster_id >= 0) {
      uniq.emplace(cluster_id, core_id);
    }
  }

  closedir(dir);
  return static_cast<unsigned>(uniq.size());
}
#endif

#ifdef _WIN32
unsigned physical_cores_winapi() {
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  std::vector<uint8_t> buf(len);
  if (!GetLogicalProcessorInformationEx(
          RelationProcessorCore,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
          &len))
    return 0;
  unsigned cnt = 0;
  for (uint8_t *p = buf.data(); p < buf.data() + len;) {
    auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
    if (info->Relationship == RelationProcessorCore) ++cnt;
    p += info->Size;
  }
  return cnt;
}
#endif

// --------------------------------------------------------------------------
// Public wrapper (all debug prints consolidated here)
// --------------------------------------------------------------------------
int getPhysicalCoreCount(int debug = 0) {
  const auto debug_print = [debug](const char *method, unsigned val) {
    if (debug > 1) std::cout << "[FINUFFT_PLAN_T] " << method << " cores=" << val << "\n";
  };
#ifdef _WIN32
  unsigned n = physical_cores_windows();
  if (n) {
    debug_print("cpuid_win", n);
    return int(n);
  }
  n = physical_cores_winapi();
  if (n) {
    debug_print("winapi", n);
    return int(n);
  }
#elif defined(__APPLE__)
  unsigned n = physical_cores_posix();
  if (n) {
    debug_print("cpuid_posix", n);
    return int(n);
  }
  n = physical_cores_sysctl();
  if (n) {
    debug_print("sysctl", n);
    return int(n);
  }
#elif defined(__linux__)
  unsigned n = physical_cores_posix();
  if (n) {
    debug_print("cpuid_posix", n);
    return int(n);
  }
  n = physical_cores_sysfs();
  if (n) {
    debug_print("sysfs", n);
    return int(n);
  }
#else
  unsigned n = 0;
#endif
  debug_print("OMP_fallback", MY_OMP_GET_MAX_THREADS());
  return MY_OMP_GET_MAX_THREADS();
}

// --------------------------------------------------------------------------
// Allowed cores (process affinity)
// --------------------------------------------------------------------------
int getAllowedCoreCount() {
#ifdef _WIN32
  DWORD_PTR pm = 0, sm = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return 0;
  int cnt = 0;
  while (pm) {
    cnt += int(pm & 1);
    pm >>= 1;
  }
  return cnt;
#elif defined(__APPLE__)
  return getPhysicalCoreCount();
#elif defined(__linux__)
  cpu_set_t cs;
  CPU_ZERO(&cs);
  if (sched_getaffinity(0, sizeof(cs), &cs) != 0) return 0;
  int cnt = 0;
  for (int i = 0; i < CPU_SETSIZE; ++i)
    if (CPU_ISSET(i, &cs)) ++cnt;
  return cnt;
#else
  return getPhysicalCoreCount();
#endif
}

} // namespace

int getOptimalThreadCount(int debug = 0) {
  if (const auto v = std::getenv("OMP_NUM_THREADS")) {
    try {
      return std::stoi(v);
    } catch (...) {
      std::cerr << "[FINUFFT_PLAN_T] OMP_NUM_THREADS env var is not a valid integer: "
                << v << "\n";
      std::cerr << "[FINUFFT_PLAN_T] Using default thread count instead.\n";
    }
  }
  const auto physical_threads = std::max(0, getPhysicalCoreCount(debug));
  const auto allowed_threads  = std::max(0, getAllowedCoreCount());
  auto optimal                = std::min(physical_threads, allowed_threads);
  if (optimal == 0) optimal = MY_OMP_GET_MAX_THREADS();
  return optimal;
}

#endif

#ifdef _WIN32
int rand_r(unsigned *seedp) {
  std::mt19937_64 gen(*seedp);
  *seedp = gen();
  return int(gen() & 0x7FFFFFFF);
}
#endif

} // namespace finufft::utils
