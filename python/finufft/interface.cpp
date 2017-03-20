#include <vector>
#include <complex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "finufft.h"

namespace py = pybind11;
typedef std::complex<double> complex_t;

py::array_t<complex_t> nufft1d1(py::array_t<double> x, py::array_t<complex_t> c, int n_modes, double accuracy, int debug) {
  auto buf_x = x.request(), buf_c = c.request();

  if (buf_x.ndim != 1 || buf_c.ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");
  if (buf_x.size != buf_c.size)
    throw std::runtime_error("Input shapes must match");
  long M = buf_x.size;

  nufft_opts opts;
  opts.debug = debug;
  auto result = py::array_t<complex_t>(n_modes);
  auto buf_F = result.request();
  int ier = finufft1d1(M, (double*)buf_x.ptr, (complex_t*)buf_c.ptr, +1, accuracy, n_modes, (complex_t*)buf_F.ptr, opts);
  if (ier != 0) {
    std::ostringstream msg;
    msg << "finufft1d1 failed with code " << ier;
    throw std::runtime_error(msg.str());
  }

  return result;
}

PYBIND11_PLUGIN(interface) {
  py::module m("interface", R"delim(
Docs
)delim");

  m.def("nufft1d1", &nufft1d1, R"delim(
Docs
)delim");

  return m.ptr();
}
