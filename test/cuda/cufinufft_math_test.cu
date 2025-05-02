#include <complex>
#include <iostream>
#include <limits>
#include <random>

// Include the custom operators for cuComplex
#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/types.h>

// Helper function to create cuComplex
template<typename T> cuda_complex<T> make_cuda_complex(T real, T imag) {
  return cuda_complex<T>{real, imag};
}

// Helper function to compare cuComplex with std::complex<T> using 1 - ratio as error
template<typename T>
bool compareComplexRel(const cuda_complex<T> a, const std::complex<T> b,
                       const std::string &operation,
                       T epsilon = std::numeric_limits<T>::epsilon()) {
  const auto std_a = std::complex<T>(a.x, a.y);
  const auto err   = std::abs(std_a - b) / std::abs(std_a);
  const auto tol   = epsilon * T(10); // factor to allow for rounding error
  if (err > tol) {
    std::cout << "Comparison failed in operation: " << operation << "\n";
    std::cout << "cuComplex: (" << a.x << ", " << a.y << ")\n";
    std::cout << "std::complex: (" << b.real() << ", " << b.imag() << ")\n";
    std::cout << "RelError: " << err << "\n";
  }
  return err <= tol;
}

template<typename T>
bool compareComplexAbs(const cuda_complex<T> a, const std::complex<T> b,
                       const std::string &operation,
                       T epsilon = std::numeric_limits<T>::epsilon()) {
  const auto std_a = std::complex<T>(a.x, a.y);
  const auto err   = std::abs(std_a - b);
  const auto tol   = epsilon * T(10); // factor to allow for rounding error
  if (err > tol) {
    std::cout << "Comparison failed in operation: " << operation << "\n";
    std::cout << "cuComplex: (" << a.x << ", " << a.y << ")\n";
    std::cout << "std::complex: (" << b.real() << ", " << b.imag() << ")\n";
    std::cout << "AbsError: " << err << "\n";
  }
  return err <= tol;
}

template<typename T> int testRandomOperations() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(-1.0, 1.0);

  for (int i = 0; i < 1000; ++i) {
    T real1  = dis(gen);
    T imag1  = dis(gen);
    T real2  = dis(gen);
    T imag2  = dis(gen);
    T scalar = dis(gen);

    cuda_complex<T> a = make_cuda_complex(real1, imag1);
    cuda_complex<T> b = make_cuda_complex(real2, imag2);
    std::complex<T> std_a(real1, imag1);
    std::complex<T> std_b(real2, imag2);

    // Test addition
    cuda_complex<T> result_add   = a + b;
    std::complex<T> expected_add = std_a + std_b;
    if (!compareComplexAbs(result_add, expected_add,
                           "add complex<" + std::string(typeid(T).name()) + "> complex<" +
                               std::string(typeid(T).name()) + ">"))
      return 1;

    // Test subtraction
    cuda_complex<T> result_sub   = a - b;
    std::complex<T> expected_sub = std_a - std_b;
    if (!compareComplexAbs(result_sub, expected_sub,
                           "sub complex<" + std::string(typeid(T).name()) + "> complex<" +
                               std::string(typeid(T).name()) + ">"))
      return 1;

    // Test multiplication
    cuda_complex<T> result_mul   = a * b;
    std::complex<T> expected_mul = std_a * std_b;
    if (!compareComplexRel(result_mul, expected_mul,
                           "mul complex<" + std::string(typeid(T).name()) + "> complex<" +
                               std::string(typeid(T).name()) + ">"))
      return 1;

    // Test division
    cuda_complex<T> result_div   = a / b;
    std::complex<T> expected_div = std_a / std_b;
    if (!compareComplexRel(result_div, expected_div,
                           "div complex<" + std::string(typeid(T).name()) + "> complex<" +
                               std::string(typeid(T).name()) + ">"))
      return 1;

    // Test addition with scalar
    cuda_complex<T> result_add_scalar   = a + scalar;
    std::complex<T> expected_add_scalar = std_a + scalar;
    if (!compareComplexRel(result_add_scalar, expected_add_scalar,
                           "add complex<" + std::string(typeid(T).name()) + "> scalar<" +
                               std::string(typeid(T).name()) + ">"))
      return 1;

    // Test subtraction with scalar
    cuda_complex<T> result_sub_scalar   = a - scalar;
    std::complex<T> expected_sub_scalar = std_a - scalar;
    if (!compareComplexRel(result_sub_scalar, expected_sub_scalar,
                           "sub complex<" + std::string(typeid(T).name()) + "> scalar<" +
                               std::string(typeid(T).name()) + ">"))
      return 1;

    // Test multiplication with scalar
    cuda_complex<T> result_mul_scalar   = a * scalar;
    std::complex<T> expected_mul_scalar = std_a * scalar;
    if (!compareComplexRel(result_mul_scalar, expected_mul_scalar,
                           "mul complex<" + std::string(typeid(T).name()) + "> scalar<" +
                               std::string(typeid(T).name()) + ">"))
      return 1;

    cuda_complex<T> result_div_scalar   = a / scalar;
    std::complex<T> expected_div_scalar = std_a / scalar;
    if (!compareComplexRel(result_div_scalar, expected_div_scalar,
                           "div complex<" + std::string(typeid(T).name()) + "> scalar<" +
                               std::string(typeid(T).name()) + ">"))
      return 1;
  }
  return 0;
}

int main() {
  if (testRandomOperations<float>()) return 1;
  if (testRandomOperations<double>()) return 1;

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
