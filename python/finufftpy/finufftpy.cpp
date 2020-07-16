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
#include "utils_precindep.h"
#include "defs.h"

#undef FLT
#undef FLTf
#undef CPX
#undef CPXf
#define FLT double
#define FLTf float
#define CPX COMPLEXIFY(FLT)
#define CPXf COMPLEXIFY(FLTf)

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

// pyfinufft_plan and pyfinufftf_plan struct holding finufft_plan
// this is needed to make sure C++ has the ownership of finufft_plan
// if we directly expose finufft_plan in pybind11 module def, the ownership is at python side
typedef struct pyfinufft_plan{
    finufft_plan fp;
    pyfinufft_plan(){
        fp = NULL;
    }
} pyfinufft_plan;
typedef struct pyfinufftf_plan{
    finufftf_plan fp;
    pyfinufftf_plan(){
        fp = NULL;
    }
} pyfinufftf_plan;


void default_opts(nufft_opts &o){
    finufft_default_opts(&o);
}


// double precision guru funcs
int makeplan(int type, int n_dims, py::array_t<BIGINT> n_modes, int iflag, int n_trans,
     FLT tol, pyfinufft_plan &plan, nufft_opts &o){
    return finufft_makeplan(type,n_dims,n_modes.mutable_data(),iflag,n_trans,tol,&(plan.fp),&o);
}

int setpts(pyfinufft_plan &plan, BIGINT M, py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<FLT> zj, 
     BIGINT N, py::array_t<FLT> s, py::array_t<FLT> t, py::array_t<FLT> u){
    return finufft_setpts(plan.fp,M,xj.mutable_data(),yj.mutable_data(),zj.mutable_data(),N,s.mutable_data(),t.mutable_data(),u.mutable_data());
}

int execute(pyfinufft_plan &plan, py::array_t<CPX> weights, py::array_t<CPX> result){
    return finufft_exec(plan.fp,weights.mutable_data(),result.mutable_data());
}

int destroy(pyfinufft_plan &plan){
    return finufft_destroy(plan.fp);
}

// single precision guru funcs
int makeplanf(int type, int n_dims, py::array_t<BIGINT> n_modes, int iflag, int n_trans,
     FLTf tol, pyfinufftf_plan &plan, nufft_opts &o){
    return finufftf_makeplan(type,n_dims,n_modes.mutable_data(),iflag,n_trans,tol,&(plan.fp),&o);
}

int setptsf(pyfinufftf_plan &plan, BIGINT M, py::array_t<FLTf> xj, py::array_t<FLTf> yj, py::array_t<FLTf> zj, 
     BIGINT N, py::array_t<FLTf> s, py::array_t<FLTf> t, py::array_t<FLTf> u){
    return finufftf_setpts(plan.fp,M,xj.mutable_data(),yj.mutable_data(),zj.mutable_data(),N,s.mutable_data(),t.mutable_data(),u.mutable_data());
}

int executef(pyfinufftf_plan &plan, py::array_t<CPXf> weights, py::array_t<CPXf> result){
    return finufftf_exec(plan.fp,weights.mutable_data(),result.mutable_data());
}

int destroyf(pyfinufftf_plan &plan){
    return finufftf_destroy(plan.fp);
}


// help funcs
// double precision help funcs
int get_type(pyfinufft_plan &plan){
    return plan.fp->type;
}

int get_dim(pyfinufft_plan &plan){
    return plan.fp->dim;
}

BIGINT get_nj(pyfinufft_plan &plan){
    return plan.fp->nj;
}

BIGINT get_nk(pyfinufft_plan &plan){
    return plan.fp->nk;
}

py::tuple get_nmodes(pyfinufft_plan &plan){
    BIGINT ms = plan.fp->ms ? plan.fp->ms : 1;
    BIGINT mt = plan.fp->mt ? plan.fp->mt : 1;
    BIGINT mu = plan.fp->mu ? plan.fp->mu : 1;
    if(plan.fp->dim<3) mu = 1;
    if(plan.fp->dim<2) mt = 1;
    return py::make_tuple(ms,mt,mu);
}

int get_ntrans(pyfinufft_plan &plan){
    return plan.fp->ntrans;
}


// single precision help funcs
int get_typef(pyfinufftf_plan &plan){
    return plan.fp->type;
}

int get_dimf(pyfinufftf_plan &plan){
    return plan.fp->dim;
}

BIGINT get_njf(pyfinufftf_plan &plan){
    return plan.fp->nj;
}

BIGINT get_nkf(pyfinufftf_plan &plan){
    return plan.fp->nk;
}

py::tuple get_nmodesf(pyfinufftf_plan &plan){
    BIGINT ms = plan.fp->ms ? plan.fp->ms : 1;
    BIGINT mt = plan.fp->mt ? plan.fp->mt : 1;
    BIGINT mu = plan.fp->mu ? plan.fp->mu : 1;
    if(plan.fp->dim<3) mu = 1;
    if(plan.fp->dim<2) mt = 1;
    return py::make_tuple(ms,mt,mu);
}

int get_ntransf(pyfinufftf_plan &plan){
    return plan.fp->ntrans;
}

PYBIND11_MODULE(finufftpy_cpp, m) {
    m.doc() = "pybind11 finufft plugin"; // optional module docstring

    // functions
    m.def("default_opts", &default_opts, "Set default nufft opts");
    m.def("makeplan", &makeplan, "Make finufft double precision plan");
    m.def("setpts", &setpts, "Set points for double precision");
    m.def("execute", &execute, "Execute for double precision");
    m.def("destroy", &destroy, "Destroy for double precision");
    m.def("makeplanf", &makeplanf, "Make finufft single precision plan");
    m.def("setptsf", &setptsf, "Set points for single precision");
    m.def("executef", &executef, "Execute for single precision");
    m.def("destroyf", &destroyf, "Destroy for single precision");
    m.def("get_type", &get_type, "Get FINUFFT transfer type");
    m.def("get_dim", &get_dim, "Get FINUFFT dimension");
    m.def("get_nj", &get_nj, "Get FINUFFT number of nu points");
    m.def("get_nk", &get_nk, "Get FINUFFT number of nu modes for type 3");
    m.def("get_nmodes", &get_nmodes, "Get FINUFFT nmodes for type 1 and type 2");
    m.def("get_ntrans", &get_ntrans, "Get FINUFFT ntrans");
    m.def("get_typef", &get_typef, "Get FINUFFT transfer type");
    m.def("get_dimf", &get_dimf, "Get FINUFFT dimension");
    m.def("get_njf", &get_njf, "Get FINUFFT number of nu points");
    m.def("get_nkf", &get_nkf, "Get FINUFFT number of nu modes for type 3");
    m.def("get_nmodesf", &get_nmodesf, "Get FINUFFT nmodes for type 1 and type 2");
    m.def("get_ntransf", &get_ntransf, "Get FINUFFT ntrans");


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
        .def_readwrite("maxbatchsize", &nufft_opts::maxbatchsize)
        .def_readwrite("showwarn", &nufft_opts::showwarn);

    // finufft_plan stuct for double precision
    py::class_<pyfinufft_plan>(m,"finufft_plan")
        .def(py::init<>());

    // finufft_plan stuct for single precision
    py::class_<pyfinufftf_plan>(m,"finufftf_plan")
        .def(py::init<>());
}
