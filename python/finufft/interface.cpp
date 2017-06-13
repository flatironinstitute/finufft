#include <vector>
#include <string>
#include <complex>
#include <exception>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "dirft.h"
#include "finufft.h"

namespace py = pybind11;

// A custom error handler to propagate errors back to Python
class finufft_error : public std::exception {
public:
  finufft_error (const std::string& msg) : msg_(msg) {};
  virtual const char* what() const throw() {
    return msg_.c_str();
  }
private:
  std::string msg_;
};

// A few macros to help reduce copying


// The interface functions:
#define ASSEMBLE_OPTIONS               \
  nufft_opts opts;                     \
  opts.R = R;                          \
  opts.debug = debug;                  \
  opts.spread_debug = spread_debug;    \
  opts.spread_sort = spread_sort;      \
  opts.fftw = !fftw ? FFTW_ESTIMATE : FFTW_MEASURE;  // ahb; def in fftw3.h

#define CHECK_FLAG                                  \
  if (ier != 0) {                                   \
    std::ostringstream msg;                         \
    msg << "finufft1d1 failed with code " << ier;   \
    throw finufft_error(msg.str());                 \
  }

// ------------
// 1D INTERFACE
// ------------

py::array_t<CPX> nufft1d1(
  py::array_t<FLT> xj, py::array_t<CPX> cj,
  INT ms,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, int fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj and cj must be 1-dimensional");
  if (buf_x.size != buf_c.size)
    throw finufft_error("xj and cj must be the same length");
  long n = buf_x.size;

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>(ms);
  auto buf_F = result.request();

  // Run the driver
  int ier = finufft1d1(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, eps, ms, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG
  return result;
}


py::array_t<CPX> nufft1d2(
  py::array_t<FLT> xj, py::array_t<CPX> fk,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, int fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_F = fk.request();
  if (buf_x.ndim != 1 || buf_F.ndim != 1)
    throw finufft_error("xj and fk must be 1-dimensional");
  long n = buf_x.size, ms = buf_F.size;

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>(n);
  auto buf_c = result.request();

  // Run the driver
  int ier = finufft1d2(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, eps, ms, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG
  return result;
}

py::array_t<CPX> nufft1d3(
  py::array_t<FLT> xj, py::array_t<CPX> cj,
  py::array_t<FLT> s,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, int fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj and cj must be 1-dimensional");
  if (buf_x.size != buf_c.size)
    throw finufft_error("xj and cj must be the same length");
  long n = buf_x.size;

  auto buf_s = s.request();
  if (buf_s.ndim != 1)
    throw finufft_error("s must be 1-dimensional");
  long nk = buf_s.size;

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>(nk);
  auto buf_F = result.request();

  // Run the driver
  int ier = finufft1d3(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, eps, nk, (FLT*)buf_s.ptr, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG
  return result;
}

// ----------------
// DIRECT INTERFACE
// ----------------

py::array_t<CPX> dirft1d1_(
  py::array_t<FLT> xj, py::array_t<CPX> cj, INT ms, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj and cj must be 1-dimensional");
  if (buf_x.size != buf_c.size)
    throw finufft_error("xj and cj must be the same length");
  long n = buf_x.size;

  // Allocate output
  auto result = py::array_t<CPX>(ms);
  auto buf_F = result.request();

  // Run the driver
  dirft1d1(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, ms, (CPX*)buf_F.ptr
  );

  return result;
}

py::array_t<CPX> dirft1d2_(
  py::array_t<FLT> xj, py::array_t<CPX> fk, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_F = fk.request();
  if (buf_x.ndim != 1 || buf_F.ndim != 1)
    throw finufft_error("xj and fk must be 1-dimensional");
  long n = buf_x.size,
       ms = buf_F.size;

  // Allocate output
  auto result = py::array_t<CPX>(n);
  auto buf_c = result.request();

  // Run the driver
  dirft1d2(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, ms, (CPX*)buf_F.ptr
  );

  return result;
}

py::array_t<CPX> dirft1d3_(
  py::array_t<FLT> xj, py::array_t<CPX> cj, py::array_t<FLT> s, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_c = cj.request(),
       buf_s = s.request();
  if (buf_x.ndim != 1 || buf_c.ndim != 1 || buf_s.ndim != 1)
    throw finufft_error("xj, cj, and s must be 1-dimensional");
  if (buf_x.size != buf_c.size)
    throw finufft_error("xj and cj must be the same length");
  long n = buf_x.size,
       ms = buf_s.size;

  // Allocate output
  auto result = py::array_t<CPX>(ms);
  auto buf_F = result.request();

  // Run the driver
  dirft1d3(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, ms, (FLT*)buf_s.ptr, (CPX*)buf_F.ptr
  );

  return result;
}


PYBIND11_PLUGIN(interface) {
  py::module m("interface", R"delim(
Docs
)delim");

  m.def("nufft1d1", &nufft1d1, R"delim(
Type-1 1D complex nonuniform FFT

::

              nj-1
     fk(k1) = SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2
              j=0

Args:
    xj (float[n]): location of sources on interval [-pi, pi]
    cj (complex[n]): FLT complex array of source strengths
    ms (int): number of Fourier modes computed, may be even or odd;
        in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
    fk (complex[ms]): FLT complex array of Fourier transform values
        (increasing mode ordering)

)delim",
    py::arg("xj"),
    py::arg("cj"),
    py::arg("ms"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = 0
  );

  m.def("nufft1d2", &nufft1d2, R"delim(
Type-2 1D complex nonuniform FFT

::

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

Args:
    xj (float[n]): location of sources on interval [-pi, pi]
    fk (complex[ms]): complex FLT array of nj answers at targets
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
    cj (complex[n]): FLT complex array of source strengths

)delim",
    py::arg("xj"),
    py::arg("fk"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = 0
  );

  m.def("nufft1d3", &nufft1d3, R"delim(
Type-3 1D complex nonuniform FFT.

::

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i s[k] xj[j]),      for k = 0, ..., nk-1
               j=0

Args:
    xj (float[n]): location of sources on interval [-pi, pi]
    cj (complex[n]): FLT complex array of source strengths
    s (float[n]): frequency locations of targets in R.
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
     fk (complex[ms]): complex FLT array of nj answers at targets

)delim",
    py::arg("xj"),
    py::arg("cj"),
    py::arg("s"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = 0
  );

  // DIRECT
  m.def("dirft1d1", &dirft1d1_, "Type-1 1D direct",
    py::arg("xj"),
    py::arg("cj"),
    py::arg("ms"),
    py::arg("iflag") = 1
  );
  m.def("dirft1d2", &dirft1d2_, "Type-2 1D direct",
    py::arg("xj"),
    py::arg("fk"),
    py::arg("iflag") = 1
  );
  m.def("dirft1d3", &dirft1d3_, "Type-3 1D direct",
    py::arg("xj"),
    py::arg("fk"),
    py::arg("s"),
    py::arg("iflag") = 1
  );

  py::register_exception<finufft_error>(m, "FINUFFTError");

  return m.ptr();
}
