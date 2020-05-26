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
  /*
   ::
     Set the default controlling options
   Args:
     opts  (nufft_opts): struct controlling options
   */
    finufft_default_opts(&o);
}

int makeplan(int type, int n_dims, py::array_t<BIGINT> n_modes, int iflag, int n_transf, 
     FLT tol, finufft_plan &plan, nufft_opts &o){
  /*
   ::
     Populates the fields of finufft_plan.
     For types 1,2 allocates memory for internal working arrays,
     evaluates spreading kernel coefficients, and instantiates the fftw_plan
   Args:
     type  (int): the type of NUFFT transform
     n_dims  (int): the dimension of NUFFT transform,(1,2 or 3)
     n_modes  (int64[3]): the number of modes in each dimension
     iflag  (int): if>=0, uses + sign in exponential, otherwise - sign
     n_transf  (int): number of NUFFT transforms
     tol  (float): precision requested (>1e-16)
     plan  (finufft_plan): struct pointer for FINUFFT plan
     opts  (nufft_opts): struct controlling options
   Returns:
     int: 0 if success, else see ../docs/usage.rst
   Example:
     see ``python/tests/python_guru1d1.py``
   */
    return finufft_makeplan(type,n_dims,n_modes.mutable_data(),iflag,n_transf,tol,&plan,&o);
}

int setpts(finufft_plan &plan, BIGINT M, py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<FLT> zj, 
     BIGINT N, py::array_t<FLT> s, py::array_t<FLT> t, py::array_t<FLT> u){
  /*
   ::
     For type 1,2: just checks and sorts the NU points.
     For type 3: allocates internal working arrays, scales/centers the NU points and NU target freqs, evaluates spreading kernel FT at all target freqs.
   Args:
     plan  (finufft_plan): struct pointer for FINUFFT plan
     M  (int64): number of nonuniform points
     xj  (float[M]): nonuniform source point x-coords
     yj  (float[M]): nonuniform source point y-coords
     zj  (float[M]): nonuniform source point z-coords
     N  (int64): number of target frequency points for type 3
     s  (float[N]): nonuniform target x-frequencies for type 3
     t  (float[N]): nonuniform target y-frequencies for type 3
     u  (float[N]): nonuniform target z-frequencies for type 3
   */
    return finufft_setpts(&plan,M,xj.mutable_data(),yj.mutable_data(),zj.mutable_data(),N,s.mutable_data(),t.mutable_data(),u.mutable_data());
}

int execute(finufft_plan &plan, py::array_t<CPX> weights, py::array_t<CPX> result){
  /*
   ::
   Args:
     weights  : source strengths
     result   : Fourier mode coefficients
   */
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
          .def_readwrite("spread_thread", &nufft_opts::spread_thread)
          .def_readwrite("maxbatchsize", &nufft_opts::maxbatchsize);

      // finufft_plan stuct
      py::class_<finufft_plan>(m,"finufft_plan")
          .def(py::init<>());
}
