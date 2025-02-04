
#include <finufft/heuristics.hpp>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

//---------------------------------------------------------------------
// Test case structure.
template<typename T> struct TestCase {
  std::string description;
  int nx, ny, nz; // Grid sizes (used to compute volume and dimension)
  int numPts;
  int nufftType;
  double epsilon; // Epsilon is always double.
  double expectedUpsample;
};

// Helper to compute dimension from ny and nz.
int computeDim(int ny, int nz) { return 1 + (ny > 1) + (nz > 1); }
//---------------------------------------------------------------------
// Run test cases for float precision.
int runTestsFloat() {
  std::vector<TestCase<float>> tests = {
      // NUFFT Type 1, 1D (density ~10) [volume = 1e6]
      {"Type1 1D, float32, density=10, ε=1e-01", 1000000, 1, 1, 10000000, 1, 1e-01, 1.25},
      {"Type1 1D, float32, density=10, ε=1e-03", 1000000, 1, 1, 10000000, 1, 1e-03, 2.0},
      {"Type1 1D, float32, density=10, ε=1e-04", 1000000, 1, 1, 10000000, 1, 1e-04, 1.25},

      // NUFFT Type 1, 2D normal density (1000×1000, density ~10)
      {"Type1 2D, float32, normal, ε=1e-01", 1000, 1000, 1, 10000000, 1, 1e-01, 1.25},
      {"Type1 2D, float32, normal, ε=1e-05", 1000, 1000, 1, 10000000, 1, 1e-05, 2.0},

      // NUFFT Type 1, 2D high density (50×50, density = 10000000/2500 = 4000)
      {"Type1 2D, float32, high, ε=1e-01", 50, 50, 1, 10000000, 1, 1e-01, 1.25},
      {"Type1 2D, float32, high, ε=1e-05", 50, 50, 1, 10000000, 1, 1e-05, 2.0},
      {"Type1 2D, float32, high, ε=1e-06", 50, 50, 1, 10000000, 1, 1e-06, 2.0},

      // NUFFT Type 1, 3D normal density (100×100×100, density ~10)
      {"Type1 3D, float32, normal, ε=1e-05", 100, 100, 100, 10000000, 1, 1e-05, 2.0},

      // NUFFT Type 1, 1D very low density (density = 10000000/20000000 = 0.5)
      {"Type1 1D, float32, very low, ε=1e-03", 20000000, 1, 1, 10000000, 1, 1e-03, 1.25},

      // NUFFT Type 2, 1D (density ~10)
      {"Type2 1D, float32, density=10, ε=1e-01", 1000000, 1, 1, 10000000, 2, 1e-01, 1.25},
      {"Type2 1D, float32, density=10, ε=1e-05", 1000000, 1, 1, 10000000, 2, 1e-05, 2.0},

      // NUFFT Type 2, 2D normal density (1000×1000, density ~10)
      {"Type2 2D, float32, normal, ε=1e-01", 1000, 1000, 1, 10000000, 2, 1e-01, 1.25},

      // NUFFT Type 2, 3D high density (20×20×20, density = 10000000/8000 = 1250)
      {"Type2 3D, float32, high, ε=1e-03", 20, 20, 20, 10000000, 2, 1e-03, 2.0},
      {"Type2 3D, float32, high, ε=1e-04", 20, 20, 20, 10000000, 2, 1e-04, 2.0}};
  int testNum = 0;

  for (const auto &test : tests) {
    testNum++;
    int volume    = test.nx * test.ny * test.nz;
    int dim       = computeDim(test.ny, test.nz);
    double result = finufft::heuristics::bestUpsamplingFactor<float>(
        volume, test.numPts, dim, test.nufftType, test.epsilon);
    if (result != test.expectedUpsample) {
      std::cerr << "Float Test #" << testNum << " failed: " << test.description
                << "\n  (volume=" << volume
                << ", density=" << double(test.numPts) / volume << ", dim=" << dim
                << "), nufftType=" << test.nufftType << ", ε=" << test.epsilon
                << "\n  Expected " << test.expectedUpsample << " but got " << result
                << "\n";
      return 1;
    } else {
      std::cout << "Float Test #" << testNum << " passed: " << test.description << "\n";
    }
  }
  return 0;
}

//---------------------------------------------------------------------
// Run test cases for double precision.
int runTestsDouble() {
  std::vector<TestCase<double>> tests = {
      // NUFFT Type 1, 1D (density ~10)
      {"Type1 1D, float64, density=10, ε=1e-01", 1000000, 1, 1, 10000000, 1, 1e-01, 1.25},
      {"Type1 1D, float64, density=10, ε=1e-03", 1000000, 1, 1, 10000000, 1, 1e-03, 1.25},

      // NUFFT Type 1, 2D normal density (1000×1000, density ~10)
      {"Type1 2D, float64, normal, ε=1e-01", 1000, 1000, 1, 10000000, 1, 1e-01, 1.25},
      {"Type1 2D, float64, normal, ε=1e-07", 1000, 1000, 1, 10000000, 1, 1e-07, 2.0},
      {"Type1 2D, float64, normal, ε=1e-08", 1000, 1000, 1, 10000000, 1, 1e-08, 2.0},
      {"Type1 2D, float64, normal, ε=1e-09", 1000, 1000, 1, 10000000, 1, 1e-09, 2.0},

      // NUFFT Type 1, 2D high density (50×50, density = 4000)
      {"Type1 2D, float64, high, ε=1e-01", 50, 50, 1, 10000000, 1, 1e-01, 1.25},
      {"Type1 2D, float64, high, ε=1e-05", 50, 50, 1, 10000000, 1, 1e-05, 2.0},

      // NUFFT Type 1, 3D normal density (100×100×100, density ~10)
      {"Type1 3D, float64, normal, ε=1e-01", 100, 100, 100, 10000000, 1, 1e-01, 1.25},
      {"Type1 3D, float64, normal, ε=1e-03", 100, 100, 100, 10000000, 1, 1e-03, 1.25},
      {"Type1 3D, float64, normal, ε=1e-04", 100, 100, 100, 10000000, 1, 1e-04, 2.0},
      {"Type1 3D, float64, normal, ε=1e-05", 100, 100, 100, 10000000, 1, 1e-05, 2.0},
      {"Type1 3D, float64, normal, ε=1e-06", 100, 100, 100, 10000000, 1, 1e-06, 2.0},
      {"Type1 3D, float64, normal, ε=1e-07", 100, 100, 100, 10000000, 1, 1e-07, 2.0},
      {"Type1 3D, float64, normal, ε=1e-08", 100, 100, 100, 10000000, 1, 1e-08, 2.0},
      {"Type1 3D, float64, normal, ε=1e-09", 100, 100, 100, 10000000, 1, 1e-09, 2.0},

      // NUFFT Type 1, 3D high density (20×20×20, density = 10000000/8000 = 1250)
      {"Type1 3D, float64, high, ε=1e-01", 20, 20, 20, 10000000, 1, 1e-01, 1.25},
      {"Type1 3D, float64, high, ε=1e-02", 20, 20, 20, 10000000, 1, 1e-02, 2.0},
      {"Type1 3D, float64, high, ε=1e-03", 20, 20, 20, 10000000, 1, 1e-03, 2.0},
      {"Type1 3D, float64, high, ε=1e-04", 20, 20, 20, 10000000, 1, 1e-04, 2.0},

      // NUFFT Type 2, 1D (density ~10)
      {"Type2 1D, float64, density=10, ε=1e-01", 1000000, 1, 1, 10000000, 2, 1e-01, 1.25},
      {"Type2 1D, float64, density=10, ε=1e-09", 1000000, 1, 1, 10000000, 2, 1e-09, 2.0},

      // NUFFT Type 2, 2D normal density (1000×1000, density ~10)
      {"Type2 2D, float64, normal, ε=1e-08", 1000, 1000, 1, 10000000, 2, 1e-08, 2.0},

      // NUFFT Type 2, 3D normal density (100×100×100, density ~10)
      {"Type2 3D, float64, normal, ε=1e-01", 100, 100, 100, 10000000, 2, 1e-01, 1.25},
      {"Type2 3D, float64, normal, ε=1e-03", 100, 100, 100, 10000000, 2, 1e-03, 2.0},

      // NUFFT Type 2, 3D high density (20×20×20, density = 1250)
      {"Type2 3D, float64, high, ε=1e-01", 20, 20, 20, 10000000, 2, 1e-01, 1.25},
      {"Type2 3D, float64, high, ε=1e-02", 20, 20, 20, 10000000, 2, 1e-02, 2.0}};

  int testNum = 0;
  for (const auto &test : tests) {
    testNum++;
    int volume    = test.nx * test.ny * test.nz;
    int dim       = computeDim(test.ny, test.nz);
    double result = finufft::heuristics::bestUpsamplingFactor<double>(
        volume, test.numPts, dim, test.nufftType, test.epsilon);
    if (result != test.expectedUpsample) {
      std::cerr << "Double Test #" << testNum << " failed: " << test.description
                << "\n  (volume=" << volume
                << ", density=" << double(test.numPts) / volume << ", dim=" << dim
                << "), nufftType=" << test.nufftType << ", ε=" << test.epsilon
                << "\n  Expected " << test.expectedUpsample << " but got " << result
                << "\n";
      return 1;
    } else {
      std::cout << "Double Test #" << testNum << " passed: " << test.description << "\n";
    }
  }
  return 0;
}

int main() {
  int errFloat  = runTestsFloat();
  int errDouble = runTestsDouble();
  if (errFloat != 0 || errDouble != 0) {
    std::cerr << "\nOne or more tests failed.\n";
    return 1;
  }
  std::cout << "\nAll tests passed.\n";
  return 0;
}
