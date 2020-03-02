// python interface to C++ wrappers to FINUFFT.
// Dan Foreman-Mackey, Jeremy Magland, and Alex Barnett.
//
// Warning: users should not call the below-defined routines
// finufftpy.finufftpy_cpp.* from python.
// Rather, they should call finufftpy.nufftpy?d?  which are documented in
// _interfaces.py
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>
#include "finufft.h"
#include "utils.h"

namespace py = pybind11;

// DFM's custom error handler to propagate errors back to Python
class error : public std::exception {
public:
  error (const std::string& msg) : msg_(msg) {};
  virtual const char* what() const throw() {
    return msg_.c_str();
  }
private:
  std::string msg_;
};

static int fftwoptslist[] = {FFTW_ESTIMATE,FFTW_MEASURE,FFTW_PATIENT,FFTW_EXHAUSTIVE};

// 0, finufft_default_opts(&opts)
// 1, finufft_makeplane
// 2, finufft_setpts
// 3, finufft_exec(&plan,c,F)
// 4, finufft_destroy
// 5, fftwopts
// 6, get_max_threads

void default_opts(nufft_opts &o){
    finufft_default_opts(&o);
}

int makeplan(int type, int n_dims, py::array_t<BIGINT> n_modes, int iflag, int n_transf, 
     FLT tol, int blksize, finufft_plan &plan, nufft_opts &o){
    return finufft_makeplan(type,n_dims,n_modes.mutable_data(),iflag,n_transf,tol,blksize,&plan,&o);
}

int setpts(finufft_plan &plan, BIGINT M, py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<FLT> zj, 
     BIGINT N, py::array_t<FLT> s, py::array_t<FLT> t, py::array_t<FLT> u){
    return finufft_setpts(&plan,M,xj.mutable_data(),yj.mutable_data(),zj.mutable_data(),N,s.mutable_data(),t.mutable_data(),u.mutable_data());
}

int execute(finufft_plan &plan, py::array_t<CPX> weights, py::array_t<CPX> result){
    return finufft_exec(&plan,weights.mutable_data(),result.mutable_data());
}

int destroy(finufft_plan &plan){
    return finufft_destroy(&plan);
}

int fftwopts(int fftw){
    if(fftw<0||fftw>3)
        throw error("invalid fftw option value");
    return fftwoptslist[fftw];
}

int get_max_threads(){
    return MY_OMP_GET_MAX_THREADS();
}

PYBIND11_MODULE(finufftpy_cpp, m) {
      m.doc() = "pybind11 finufft plugin"; // optional module docstring

      // functions
      m.def("default_opts", &default_opts, "Set default nufft opts");
      m.def("makeplan", &makeplan, "Make finufft plan");
      m.def("setpts", &setpts, "Set points");
      m.def("execute", &execute, "Execute");
      m.def("destroy", &destroy, "Destroy");
      m.def("fftwopts", &fftwopts, "FFTW options");
      m.def("get_max_threads", &get_max_threads, "Get max number of threads");

      // nufft_opts struct
      py::class_<nufft_opts>(m,"nufft_opts")
          .def(py::init<>())
          .def_readwrite("debug", &nufft_opts::debug)
          .def_readwrite("spread_debug", &nufft_opts::spread_debug)
          .def_readwrite("spread_sort", &nufft_opts::spread_sort)
          .def_readwrite("spread_kerevalmeth", &nufft_opts::spread_kerevalmeth)
          .def_readwrite("spread_kerpad", &nufft_opts::spread_kerpad)
          .def_readwrite("chkbnds", &nufft_opts::chkbnds)
          .def_readwrite("fftw", &nufft_opts::fftw)
          .def_readwrite("modeord", &nufft_opts::modeord)
          .def_readwrite("upsampfac", &nufft_opts::upsampfac)
          .def_readwrite("spread_scheme", &nufft_opts::spread_scheme);

      // finufft_plan stuct
      py::class_<finufft_plan>(m,"finufft_plan")
          .def(py::init<>());
}
