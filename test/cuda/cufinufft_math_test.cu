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
bool compareComplex(const cuda_complex<T> &a, const std::complex<T> &b,
                    const std::string &operation,
                    T epsilon = std::numeric_limits<T>::epsilon()) {
  T real_error = 1 - a.x / b.real();
  T imag_error = 1 - a.y / b.imag();
  if (real_error >= epsilon || imag_error >= epsilon) {
    std::cout << "Comparison failed in operation: " << operation << "\n";
    std::cout << "cuComplex: (" << a.x << ", " << a.y << ")\n";
    std::cout << "std::complex: (" << b.real() << ", " << b.imag() << ")\n";
    std::cout << "Real error: " << real_error << "\n";
    std::cout << "Imag error: " << imag_error << "\n";
  }
  return real_error < epsilon && imag_error < epsilon;
}

template<typename T> int testRandomOperations() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(-100.0, 100.0);

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
    if (!compareComplex(result_add, expected_add,
                        "add complex<" + std::string(typeid(T).name()) + "> complex<" +
                            std::string(typeid(T).name()) + ">"))
      return 1;

    // Test subtraction
    cuda_complex<T> result_sub   = a - b;
    std::complex<T> expected_sub = std_a - std_b;
    if (!compareComplex(result_sub, expected_sub,
                        "sub complex<" + std::string(typeid(T).name()) + "> complex<" +
                            std::string(typeid(T).name()) + ">"))
      return 1;

    // Test multiplication
    cuda_complex<T> result_mul   = a * b;
    std::complex<T> expected_mul = std_a * std_b;
    if (!compareComplex(result_mul, expected_mul,
                        "mul complex<" + std::string(typeid(T).name()) + "> complex<" +
                            std::string(typeid(T).name()) + ">"))
      return 1;

    // Test division
    // Avoid division by small numbers as the implementation is slightly different
    // Maybe there is a better way to test it
    if (real2 < 1.0 || imag2 < 1.0) { // Avoid division by zero
      cuda_complex<T> result_div   = a / b;
      std::complex<T> expected_div = std_a / std_b;
      if (!compareComplex(result_div, expected_div,
                          "div complex<" + std::string(typeid(T).name()) + "> complex<" +
                              std::string(typeid(T).name()) + ">",
                          std::numeric_limits<T>::epsilon() * 1000))
        return 1;
    }

    // Test addition with scalar
    cuda_complex<T> result_add_scalar   = a + scalar;
    std::complex<T> expected_add_scalar = std_a + scalar;
    if (!compareComplex(result_add_scalar, expected_add_scalar,
                        "add complex<" + std::string(typeid(T).name()) + "> scalar<" +
                            std::string(typeid(T).name()) + ">"))
      return 1;

    // Test subtraction with scalar
    cuda_complex<T> result_sub_scalar   = a - scalar;
    std::complex<T> expected_sub_scalar = std_a - scalar;
    if (!compareComplex(result_sub_scalar, expected_sub_scalar,
                        "sub complex<" + std::string(typeid(T).name()) + "> scalar<" +
                            std::string(typeid(T).name()) + ">"))
      return 1;

    // Test multiplication with scalar
    cuda_complex<T> result_mul_scalar   = a * scalar;
    std::complex<T> expected_mul_scalar = std_a * scalar;
    if (!compareComplex(result_mul_scalar, expected_mul_scalar,
                        "mul complex<" + std::string(typeid(T).name()) + "> scalar<" +
                            std::string(typeid(T).name()) + ">"))
      return 1;

    // Test division with scalar
    // Avoid division by small numbers which is not accurate
    if (scalar > (std::is_same_v<T, double> ? 1e-15 : 1e-6)) {
      cuda_complex<T> result_div_scalar   = a / scalar;
      std::complex<T> expected_div_scalar = std_a / scalar;
      if (!compareComplex(result_div_scalar, expected_div_scalar,
                          "div complex<" + std::string(typeid(T).name()) + "> scalar<" +
                              std::string(typeid(T).name()) + ">"))
        return 1;
    }
  }
  return 0;
}

int main() {
  if (testRandomOperations<float>()) return 1;
  if (testRandomOperations<double>()) return 1;

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
