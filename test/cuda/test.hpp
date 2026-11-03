#include <array>
#include <cstdio>

inline void print_cuda_status_(int type, int method, int count_NU, int count_secondary,
                               float totaltime, float exectime) {

  static constexpr std::array<const char *, 3> format = {
      "[Method %d] %d NU pts to %d U pts in %.3g s:\t%.3g NU pts/s",
      "[Method %1$d] %3$d U pts to %2$d NU pts in %4$.3g s:\t%5$.3g NU pts/s",
      "[Method %d] %d NU pts to %d NU pts in %.3g s:\t%.3g NU pts/s"};

  int total_points = type == 3 ? (count_NU + count_secondary) : count_NU;

  ::printf(format.at(type - 1), method, count_NU, count_secondary, totaltime / 1000,
           total_points / totaltime * 1000);

  ::printf("\t\t\t\t\t(exec-only throughput: %.3g NU pts/s)\n",
           total_points / exectime * 1000);
}
