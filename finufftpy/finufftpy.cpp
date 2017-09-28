#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "finufft.h"
#include "ndarray.h"

namespace py = pybind11;

// A custom error handler to propagate errors back to Python
class error : public std::exception {
public:
  error (const std::string& msg) : msg_(msg) {};
  virtual const char* what() const throw() {
    return msg_.c_str();
  }
private:
  std::string msg_;
};

int finufft1d1_cpp(py::array_t<double> xj,py::array_t<CPX> cj,int iflag,double eps,int ms,py::array_t<CPX> fk) {
    NDArray<double> xja(xj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    if (xja.size!=cja.size) {
        throw error("Inconsistent dimensions between cja and fka");
    }
    if (fka.size!=ms) {
        throw error("Incorrect size for fk");
    }
    int nj=xja.size;

    nufft_opts opts;
    opts.R = 2.0;            // kernel-dep upsampling ratio (for experts)
    opts.debug = 0;          // 0: silent, 1: text timing output, 2: spread info
    opts.spread_debug = 0;   // passed to spread_opts debug: 0,1 or 2
    opts.spread_sort = 1;    // passed to spread_opts sort: 0 or 1
    opts.fftw = FFTW_ESTIMATE;
    int ret=finufft1d1(nj,xja.ptr,cja.ptr,iflag,eps,ms,fka.ptr,opts);
    
    return ret;
}

PYBIND11_MODULE(finufftpy_cpp, m) {
    m.doc() = "Python wrapper for finufft";

    m.def("finufft1d1_cpp", &finufft1d1_cpp, "Python wrapper for 1d type 1 nufft",
        py::arg("xj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),py::arg("ms"),
        py::arg("fk").noconvert());
}
