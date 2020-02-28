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

// Create and copy over FINUFFT's options struct...
#define ASSEMBLE_OPTIONS               \
  nufft_opts opts;                     \
  finufft_default_opts(&opts);         \
  opts.debug = debug;                  \
  opts.spread_debug = spread_debug;    \
  opts.spread_sort = spread_sort;      \
  opts.fftw = fftwoptslist[fftw];      \
  opts.modeord = modeord;              \
  opts.chkbnds = chkbnds;              \
  opts.upsampfac = upsampfac;

// DFM's error status reporting...
#define CHECK_FLAG(NAME)                            \
  if (ier != 0) {                                   \
    std::ostringstream msg;                         \
    msg << #NAME << " failed with code " << ier;    \
    throw error(msg.str());                         \
  }

// 0, finufft_default_opts(&opts)
// 1, finufft_makeplane
// 2, finufft_setpts
// 3, finufft_exec(&plan,c,F)
// 4, finufft_destroy

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

////////////////////////////////////////////////////////////////// 1D
int finufft1d1_cpp(py::array_t<double> xj,py::array_t<CPX> cj,int iflag,double eps,int ms,py::array_t<CPX> fk, int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    if (xj.size() != cj.size())
        throw error("Inconsistent dimensions between xj and cj");
    if (fk.size() != ms)
        throw error("Incorrect size for fk");

    finufft_plan plan;
    ASSEMBLE_OPTIONS

    BIGINT n_modes[3];
    n_modes[0] = ms;
    n_modes[1] = 1;
    n_modes[2] = 1;
    
    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj = xj.size();
    
    int ier = finufft_makeplan(1, 1, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft1d1_makeplan)
    
    ier = finufft_setpts(&plan, nj, xj.mutable_data(), NULL, NULL, ms, NULL, NULL, NULL);
    CHECK_FLAG(finufft1d1_setpts)

    ier = finufft_exec(&plan, cj.mutable_data(), fk.mutable_data());
    CHECK_FLAG(finufft1d1_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft1d1_destroy)
    
    return ier;
}
int finufft1d2_cpp(py::array_t<double> xj,py::array_t<CPX> cj,int iflag,double eps,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    if (xj.size()!=cj.size())
        throw error("Inconsistent dimensions between xj and cj");

    finufft_plan plan;
    ASSEMBLE_OPTIONS

    int ms=fk.size();
    BIGINT n_modes[3];
    n_modes[0] = ms;
    n_modes[1] = 1;
    n_modes[2] = 1;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj = xj.size();

    int ier = finufft_makeplan(2, 1, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft1d2_makeplan)
    
    ier = finufft_setpts(&plan, nj, xj.mutable_data(), NULL, NULL, ms, NULL, NULL, NULL);
    CHECK_FLAG(finufft1d2_setpts)

    ier = finufft_exec(&plan, cj.mutable_data(), fk.mutable_data());
    CHECK_FLAG(finufft1d2_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft1d2_destroy)
    
    return ier;
}
int finufft1d3_cpp(py::array_t<double> x,py::array_t<CPX> c,int iflag,double eps,py::array_t<double> s,py::array_t<CPX> f,int debug, int spread_debug, int spread_sort, int fftw,double upsampfac)
{
    if (x.size()!=c.size())
        throw error("Inconsistent dimensions between x and c");
    if (s.size()!=f.size())
        throw error("Inconsistent dimensions between s and f");

    finufft_plan plan;
    int modeord=0; int chkbnds=0;
    ASSEMBLE_OPTIONS
    
    int nk=s.size();
    BIGINT n_modes[3];
    n_modes[0] = nk;
    n_modes[1] = 1;
    n_modes[2] = 1;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj=x.size();

    int ier = finufft_makeplan(3, 1, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft1d3_makeplan)
    
    ier = finufft_setpts(&plan, nj, x.mutable_data(), NULL, NULL, nk, s.mutable_data(), NULL, NULL);
    CHECK_FLAG(finufft1d3_setpts)

    ier = finufft_exec(&plan, c.mutable_data(), f.mutable_data());
    CHECK_FLAG(finufft1d3_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft1d3_destroy)
    
    return ier;
}

////////////////////////////////////////////////////////////////////// 2D
int finufft2d1_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<CPX> cj,int iflag,double eps,int ms,int mt,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    if ((xj.size()!=cj.size())||(yj.size()!=cj.size()))
        throw error("Inconsistent dimensions between xj or yj and cj");
    if (fk.size()!=ms*mt)
        throw error("Incorrect size for fk");

    finufft_plan plan;
    ASSEMBLE_OPTIONS

    BIGINT n_modes[3];
    n_modes[0] = ms;
    n_modes[1] = mt;
    n_modes[2] = 1;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj=xj.size();

    int ier = finufft_makeplan(1, 2, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft2d1_makeplan)
    
    ier = finufft_setpts(&plan, nj, xj.mutable_data(), yj.mutable_data(), NULL, ms*mt, NULL, NULL, NULL);
    CHECK_FLAG(finufft2d1_setpts)

    ier = finufft_exec(&plan, cj.mutable_data(), fk.mutable_data());
    CHECK_FLAG(finufft2d1_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft2d1_destroy)
    
    return ier;
}
int finufft2d1many_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<CPX> cj,int iflag,double eps,int ms,int mt,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    int ndata = cj.shape(1);
    if ((xj.size()*ndata!=cj.size())||(yj.size()*ndata!=cj.size()))
        throw error("Inconsistent dimensions between xj or yj and cj");
    if (fk.size()!=ms*mt*ndata)
        throw error("Incorrect size for fk");
    
    finufft_plan plan;
    ASSEMBLE_OPTIONS

    BIGINT n_modes[3];
    n_modes[0] = ms;
    n_modes[1] = mt;
    n_modes[2] = 1;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj = xj.size();

    int ier = finufft_makeplan(1, 2, n_modes, iflag, ndata, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft2d1many_makeplan)
    
    ier = finufft_setpts(&plan, nj, xj.mutable_data(), yj.mutable_data(), NULL, ms*mt, NULL, NULL, NULL);
    CHECK_FLAG(finufft2d1many_setpts)

    ier = finufft_exec(&plan, cj.mutable_data(), fk.mutable_data());
    CHECK_FLAG(finufft2d1many_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft2d1many_destroy)
    
    return ier;
}
int finufft2d2_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<CPX> cj,int iflag,double eps,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    if ((xj.size()!=cj.size())||(yj.size()!=cj.size()))
        throw error("Inconsistent dimensions between xj or yj and cj");

    finufft_plan plan;
    ASSEMBLE_OPTIONS

    int ms=fk.shape(0);
    int mt=fk.shape(1);
    BIGINT n_modes[3];
    n_modes[0] = ms;
    n_modes[1] = mt;
    n_modes[2] = 1;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj=xj.size();

    int ier = finufft_makeplan(2, 2, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft2d2_makeplan)
    
    ier = finufft_setpts(&plan, nj, xj.mutable_data(), yj.mutable_data(), NULL, ms*mt, NULL, NULL, NULL);
    CHECK_FLAG(finufft2d2_setpts)

    ier = finufft_exec(&plan, cj.mutable_data(), fk.mutable_data());
    CHECK_FLAG(finufft2d2_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft2d2_destroy)
    
    return ier;
}
int finufft2d2many_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<CPX> cj,int iflag,double eps,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    int ndata = fk.shape(2);
    if ((xj.size()*ndata!=cj.size())||(yj.size()*ndata!=cj.size()))
        throw error("Inconsistent dimensions between xj or yj and cj");

    finufft_plan plan;
    ASSEMBLE_OPTIONS

    int ms=fk.shape(0);
    int mt=fk.shape(1);
    BIGINT n_modes[3];
    n_modes[0] = ms;
    n_modes[1] = mt;
    n_modes[2] = 1;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj=xj.size();

    int ier = finufft_makeplan(2, 2, n_modes, iflag, ndata, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft2d2many_makeplan)
    
    ier = finufft_setpts(&plan, nj, xj.mutable_data(), yj.mutable_data(), NULL, ms*mt, NULL, NULL, NULL);
    CHECK_FLAG(finufft2d2many_setpts)

    ier = finufft_exec(&plan, cj.mutable_data(), fk.mutable_data());
    CHECK_FLAG(finufft2d2many_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft2d2many_destroy)
    
    return ier;
}
int finufft2d3_cpp(py::array_t<double> x,py::array_t<double> y,py::array_t<CPX> c,int iflag,double eps,py::array_t<double> s,py::array_t<double> t,py::array_t<CPX> f,int debug, int spread_debug, int spread_sort, int fftw,double upsampfac)
{
    if ((x.size()!=c.size())||(y.size()!=c.size()))
        throw error("Inconsistent dimensions between x or y and c");
    if ((s.size()!=f.size())||(t.size()!=f.size()))
        throw error("Inconsistent dimensions between s or t and f");

    finufft_plan plan;
    int modeord=0; int chkbnds=0;
    ASSEMBLE_OPTIONS

    int nk=s.size();
    BIGINT n_modes[3];
    n_modes[0] = nk;
    n_modes[1] = 1;
    n_modes[2] = 1;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj=x.size();

    int ier = finufft_makeplan(3, 2, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft2d3_makeplan)
    
    ier = finufft_setpts(&plan, nj, x.mutable_data(), y.mutable_data(), NULL, nk, s.mutable_data(), t.mutable_data(), NULL);
    CHECK_FLAG(finufft2d3_setpts)

    ier = finufft_exec(&plan, c.mutable_data(), f.mutable_data());
    CHECK_FLAG(finufft2d3_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft2d3_destroy)
    
    return ier;
}

////////////////////////////////////////////////////////////////////// 3D
int finufft3d1_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<double> zj,py::array_t<CPX> cj,int iflag,double eps,int ms,int mt,int mu,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    if ((xj.size()!=cj.size())||(yj.size()!=cj.size())||(zj.size()!=cj.size()))
        throw error("Inconsistent dimensions between xj or yj or zj and cj");
    if (fk.size()!=ms*mt*mu)
        throw error("Incorrect size for fk");

    finufft_plan plan;
    ASSEMBLE_OPTIONS

    BIGINT n_modes[3];
    n_modes[0] = ms;
    n_modes[1] = mt;
    n_modes[2] = mu;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj=xj.size();

    int ier = finufft_makeplan(1, 3, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft3d1_makeplan)
    
    ier = finufft_setpts(&plan, nj, xj.mutable_data(), yj.mutable_data(), zj.mutable_data(), ms*mt*mu, NULL, NULL, NULL);
    CHECK_FLAG(finufft3d1_setpts)

    ier = finufft_exec(&plan, cj.mutable_data(), fk.mutable_data());
    CHECK_FLAG(finufft3d1_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft3d1_destroy)
    
    return ier;
}
int finufft3d2_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<double> zj,py::array_t<CPX> cj,int iflag,double eps,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    if ((xj.size()!=cj.size())||(yj.size()!=cj.size())||(zj.size()!=cj.size()))
        throw error("Inconsistent dimensions between xj or yj or zj and cj");

    finufft_plan plan;
    ASSEMBLE_OPTIONS

    int ms=fk.shape(0);
    int mt=fk.shape(1);
    int mu=fk.shape(2);
    BIGINT n_modes[3];
    n_modes[0] = ms;
    n_modes[1] = mt;
    n_modes[2] = mu;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj=xj.size();
    
    int ier = finufft_makeplan(2, 3, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft3d2_makeplan)
    
    ier = finufft_setpts(&plan, nj, xj.mutable_data(), yj.mutable_data(), zj.mutable_data(), ms*mt*mu, NULL, NULL, NULL);
    CHECK_FLAG(finufft3d2_setpts)

    ier = finufft_exec(&plan, cj.mutable_data(), fk.mutable_data());
    CHECK_FLAG(finufft3d2_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft3d2_destroy)
    
    return ier;
}
int finufft3d3_cpp(py::array_t<double> x,py::array_t<double> y,py::array_t<double> z,py::array_t<CPX> c,int iflag,double eps,py::array_t<double> s,py::array_t<double> t,py::array_t<double> u,py::array_t<CPX> f,int debug, int spread_debug, int spread_sort, int fftw,double upsampfac) {
    if ((x.size()!=c.size())||(y.size()!=c.size())||(z.size()!=c.size()))
        throw error("Inconsistent dimensions between x or y or z and c");
    if ((s.size()!=f.size())||(t.size()!=f.size())||(u.size()!=f.size()))
        throw error("Inconsistent dimensions between s or t or u and f");

    finufft_plan plan;
    int modeord=0; int chkbnds=0;
    ASSEMBLE_OPTIONS

    int nk=s.size();
    BIGINT n_modes[3];
    n_modes[0] = nk;
    n_modes[1] = 1;
    n_modes[2] = 1;

    int blksize = MY_OMP_GET_MAX_THREADS();
    BIGINT nj=x.size();

    int ier = finufft_makeplan(3, 3, n_modes, iflag, 1, eps, blksize, &plan, &opts);
    CHECK_FLAG(finufft3d3_makeplan)
    
    ier = finufft_setpts(&plan, nj, x.mutable_data(), y.mutable_data(), z.mutable_data(), nk, s.mutable_data(), t.mutable_data(), u.mutable_data());
    CHECK_FLAG(finufft3d3_setpts)

    ier = finufft_exec(&plan, c.mutable_data(), f.mutable_data());
    CHECK_FLAG(finufft3d3_execute)
    
    ier = finufft_destroy(&plan);
    CHECK_FLAG(finufft3d3_destroy)
    
    return ier;
}

PYBIND11_MODULE(finufftpy_cpp, m) {
      m.doc() = "pybind11 finufft plugin"; // optional module docstring

      // functions
      m.def("default_opts", &default_opts, "Set default nufft opts");
      m.def("makeplan", &makeplan, "Make finufft plan");
      m.def("setpts", &setpts, "Set points");
      m.def("execute", &execute, "Execute");
      m.def("destroy", &destroy, "Destroy");
      m.def("finufft1d1_cpp", &finufft1d1_cpp, "Python wrapper for 1-d type 1 nufft");
      m.def("finufft1d2_cpp", &finufft1d2_cpp, "Python wrapper for 1-d type 2 nufft");
      m.def("finufft1d3_cpp", &finufft1d3_cpp, "Python wrapper for 1-d type 3 nufft");
      m.def("finufft2d1_cpp", &finufft2d1_cpp, "Python wrapper for 2-d type 1 nufft");
      m.def("finufft2d1many_cpp", &finufft2d1many_cpp, "Python wrapper for 2-d type 1 many nufft");
      m.def("finufft2d2_cpp", &finufft2d2_cpp, "Python wrapper for 2-d type 2 nufft");
      m.def("finufft2d2many_cpp", &finufft2d2many_cpp, "Python wrapper for 2-d type 2 many nufft");
      m.def("finufft2d3_cpp", &finufft2d3_cpp, "Python wrapper for 2-d type 3 nufft");
      m.def("finufft3d1_cpp", &finufft3d1_cpp, "Python wrapper for 3-d type 1 nufft");
      m.def("finufft3d2_cpp", &finufft3d2_cpp, "Python wrapper for 3-d type 2 nufft");
      m.def("finufft3d3_cpp", &finufft3d3_cpp, "Python wrapper for 3-d type 3 nufft");

      // nufft_opts struct
      py::class_<nufft_opts>(m,"nufft_opts")
          .def(py::init<>())
          //.def("set_debug",[](nufft_opts &o,int opt){o.debug=opt;})
          //.def("set_spread_debug",[](nufft_opts &o,int opt){o.spread_debug=opt;})
          //.def("set_spread_sort",[](nufft_opts &o,int opt){o.spread_sort=opt;})
          //.def("set_spread_kerevalmeth",[](nufft_opts &o,int opt){o.spread_kerevalmeth=opt;})
          //.def("set_spread_kerpad",[](nufft_opts &o,int opt){o.spread_kerpad=opt;})
          //.def("set_chkbnds",[](nufft_opts &o,int opt){o.chkbnds=opt;})
          //.def("set_fftw",[](nufft_opts &o,int opt){o.fftw=fftwoptslist[opt];})
          //.def("set_modeord",[](nufft_opts &o,int opt){o.modeord=opt;})
          //.def("set_upsampfac",[](nufft_opts &o,FLT opt){o.upsampfac=opt;})
          //.def("set_spread_scheme",[](nufft_opts &o,int opt){o.spread_scheme=opt;})
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

      // type enum
      /*
      py::enum_<finufft_type>(m,"finufft_type",py::arithmetic(),"Unscoped finufft type enumeration")
          .value("type1",type1,"type1 calculation")
          .value("type2",type2,"type2 calculation")
          .value("type3",type3,"type3 calculation")
          .export_values();
      */
}

/*
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

// basic interface to FFTW planning options, accessed in order by fftw=0,1,2....
static int fftwoptslist[] = {FFTW_ESTIMATE,FFTW_MEASURE,FFTW_PATIENT,FFTW_EXHAUSTIVE};

// Create and copy over FINUFFT's options struct...
#define ASSEMBLE_OPTIONS               \
  nufft_opts opts;                     \
  finufft_default_opts(&opts);         \
  opts.debug = debug;                  \
  opts.spread_debug = spread_debug;    \
  opts.spread_sort = spread_sort;      \
  opts.fftw = fftwoptslist[fftw];      \
  opts.modeord = modeord;              \
  opts.chkbnds = chkbnds;              \
  opts.upsampfac = upsampfac;

// DFM's error status reporting...
#define CHECK_FLAG(NAME)                            \
  if (ier != 0) {                                   \
    std::ostringstream msg;                         \
    msg << #NAME << " failed with code " << ier;    \
    throw error(msg.str());                         \
  }


////////////////////////////////////////////////////////////////// 1D

int finufft1d1_cpp(py::array_t<double> xj,py::array_t<CPX> cj,int iflag,double eps,int ms,py::array_t<CPX> fk, int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    NDArray<double> xja(xj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    if (xja.size!=cja.size)
        throw error("Inconsistent dimensions between xj and cj");
    if (fka.size!=ms)
        throw error("Incorrect size for fk");
    int nj=xja.size;
    ASSEMBLE_OPTIONS
    int ier=finufft1d1(nj,xja.ptr,cja.ptr,iflag,eps,ms,fka.ptr,opts);
    CHECK_FLAG(finufft1d1)
    return ier;
}
int finufft1d2_cpp(py::array_t<double> xj,py::array_t<CPX> cj,int iflag,double eps,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    NDArray<double> xja(xj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    if (xja.size!=cja.size)
        throw error("Inconsistent dimensions between xj and cj");
    int nj=xja.size;
    int ms=fka.size;
    ASSEMBLE_OPTIONS
    int ier=finufft1d2(nj,xja.ptr,cja.ptr,iflag,eps,ms,fka.ptr,opts);
    CHECK_FLAG(finufft1d2)
    return ier;
}
int finufft1d3_cpp(py::array_t<double> x,py::array_t<CPX> c,int iflag,double eps,py::array_t<double> s,py::array_t<CPX> f,int debug, int spread_debug, int spread_sort, int fftw,double upsampfac)
{
    NDArray<double> xa(x);
    NDArray<CPX> ca(c);
    NDArray<double> sa(s);
    NDArray<CPX> fa(f);
    if (xa.size!=ca.size)
        throw error("Inconsistent dimensions between x and c");
    if (sa.size!=fa.size)
        throw error("Inconsistent dimensions between s and f");
    int nj=xa.size;
    int nk=sa.size;
    int modeord=0; int chkbnds=0;
    ASSEMBLE_OPTIONS
    int ier=finufft1d3(nj,xa.ptr,ca.ptr,iflag,eps,nk,sa.ptr,fa.ptr,opts);
    CHECK_FLAG(finufft1d3)
    return ier;
}

////////////////////////////////////////////////////////////////////// 2D

int finufft2d1_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<CPX> cj,int iflag,double eps,int ms,int mt,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    NDArray<double> xja(xj);
    NDArray<double> yja(yj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    if ((xja.size!=cja.size)||(yja.size!=cja.size))
        throw error("Inconsistent dimensions between xj or yj and cj");
    if (fka.size!=ms*mt)
        throw error("Incorrect size for fk");
    int nj=xja.size;
    ASSEMBLE_OPTIONS
    int ier=finufft2d1(nj,xja.ptr,yja.ptr,cja.ptr,iflag,eps,ms,mt,fka.ptr,opts);
    CHECK_FLAG(finufft2d1)
    return ier;
}
int finufft2d1many_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<CPX> cj,int iflag,double eps,int ms,int mt,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    NDArray<double> xja(xj);
    NDArray<double> yja(yj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    int ndata = cja.shape[1];
    if ((xja.size*ndata!=cja.size)||(yja.size*ndata!=cja.size))
        throw error("Inconsistent dimensions between xj or yj and cj");
    if (fka.size!=ms*mt*ndata)
        throw error("Incorrect size for fk");
    int nj=xja.size;
    ASSEMBLE_OPTIONS
      int ier=finufft2d1many(ndata,nj,xja.ptr,yja.ptr,cja.ptr,iflag,eps,ms,mt,fka.ptr,opts);
    CHECK_FLAG(finufft2d1many)
    return ier;
}
int finufft2d2_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<CPX> cj,int iflag,double eps,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    NDArray<double> xja(xj);
    NDArray<double> yja(yj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    if ((xja.size!=cja.size)||(yja.size!=cja.size))
        throw error("Inconsistent dimensions between xj or yj and cj");
    int nj=xja.size;
    int ms=fka.shape[0];
    int mt=fka.shape[1];
    ASSEMBLE_OPTIONS
    int ier=finufft2d2(nj,xja.ptr,yja.ptr,cja.ptr,iflag,eps,ms,mt,fka.ptr,opts);
    CHECK_FLAG(finufft2d2)
    return ier;
}
int finufft2d2many_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<CPX> cj,int iflag,double eps,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    NDArray<double> xja(xj);
    NDArray<double> yja(yj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    int ndata = fka.shape[2];
    if ((xja.size*ndata!=cja.size)||(yja.size*ndata!=cja.size))
        throw error("Inconsistent dimensions between xj or yj and cj");
    int nj=xja.size;
    int ms=fka.shape[0];
    int mt=fka.shape[1];
    ASSEMBLE_OPTIONS
    int ier=finufft2d2many(ndata,nj,xja.ptr,yja.ptr,cja.ptr,iflag,eps,ms,mt,fka.ptr,opts);
    CHECK_FLAG(finufft2d2many)
    return ier;
}
int finufft2d3_cpp(py::array_t<double> x,py::array_t<double> y,py::array_t<CPX> c,int iflag,double eps,py::array_t<double> s,py::array_t<double> t,py::array_t<CPX> f,int debug, int spread_debug, int spread_sort, int fftw,double upsampfac)
{
    NDArray<double> xa(x);
    NDArray<double> ya(y);
    NDArray<CPX> ca(c);
    NDArray<double> sa(s);
    NDArray<double> ta(t);
    NDArray<CPX> fa(f);
    if ((xa.size!=ca.size)||(ya.size!=ca.size))
        throw error("Inconsistent dimensions between x or y and c");
    if ((sa.size!=fa.size)||(ta.size!=fa.size))
        throw error("Inconsistent dimensions between s or t and f");
    int nj=xa.size;
    int nk=sa.size;
    int modeord=0; int chkbnds=0;
    ASSEMBLE_OPTIONS
    int ier=finufft2d3(nj,xa.ptr,ya.ptr,ca.ptr,iflag,eps,nk,sa.ptr,ta.ptr,fa.ptr,opts);
    CHECK_FLAG(finufft2d3)
    return ier;
}

////////////////////////////////////////////////////////////////////// 3D

int finufft3d1_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<double> zj,py::array_t<CPX> cj,int iflag,double eps,int ms,int mt,int mu,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    NDArray<double> xja(xj);
    NDArray<double> yja(yj);
    NDArray<double> zja(zj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    if ((xja.size!=cja.size)||(yja.size!=cja.size)||(zja.size!=cja.size))
        throw error("Inconsistent dimensions between xj or yj or zj and cj");
    if (fka.size!=ms*mt*mu)
        throw error("Incorrect size for fk");
    int nj=xja.size;
    ASSEMBLE_OPTIONS
    int ier=finufft3d1(nj,xja.ptr,yja.ptr,zja.ptr,cja.ptr,iflag,eps,ms,mt,mu,fka.ptr,opts);
    CHECK_FLAG(finufft3d1)
    return ier;
}
int finufft3d2_cpp(py::array_t<double> xj,py::array_t<double> yj,py::array_t<double> zj,py::array_t<CPX> cj,int iflag,double eps,py::array_t<CPX> fk,int debug, int spread_debug, int spread_sort, int fftw, int modeord, int chkbnds,double upsampfac)
{
    NDArray<double> xja(xj);
    NDArray<double> yja(yj);
    NDArray<double> zja(zj);
    NDArray<CPX> cja(cj);
    NDArray<CPX> fka(fk);
    if ((xja.size!=cja.size)||(yja.size!=cja.size)||(zja.size!=cja.size))
        throw error("Inconsistent dimensions between xj or yj or zj and cj");
    int ms=fka.shape[0];
    int mt=fka.shape[1];
    int mu=fka.shape[2];
    int nj=xja.size;
    ASSEMBLE_OPTIONS
    int ier=finufft3d2(nj,xja.ptr,yja.ptr,zja.ptr,cja.ptr,iflag,eps,ms,mt,mu,fka.ptr,opts);
    CHECK_FLAG(finufft3d2)
    return ier;
}
int finufft3d3_cpp(py::array_t<double> x,py::array_t<double> y,py::array_t<double> z,py::array_t<CPX> c,int iflag,double eps,py::array_t<double> s,py::array_t<double> t,py::array_t<double> u,py::array_t<CPX> f,int debug, int spread_debug, int spread_sort, int fftw,double upsampfac) {
    NDArray<double> xa(x);
    NDArray<double> ya(y);
    NDArray<double> za(z);
    NDArray<CPX> ca(c);
    NDArray<double> sa(s);
    NDArray<double> ta(t);
    NDArray<double> ua(u);
    NDArray<CPX> fa(f);
    if ((xa.size!=ca.size)||(ya.size!=ca.size)||(za.size!=ca.size))
        throw error("Inconsistent dimensions between x or y or z and c");
    if ((sa.size!=fa.size)||(ta.size!=fa.size)||(ua.size!=fa.size))
        throw error("Inconsistent dimensions between s or t or u and f");
    int nj=xa.size;
    int nk=sa.size;
    int modeord=0; int chkbnds=0;
    ASSEMBLE_OPTIONS
    int ier=finufft3d3(nj,xa.ptr,ya.ptr,za.ptr,ca.ptr,iflag,eps,nk,sa.ptr,ta.ptr,ua.ptr,fa.ptr,opts);
    CHECK_FLAG(finufft3d3)
    return ier;
}


// module:


PYBIND11_MODULE(finufftpy_cpp, m) {
  m.doc() = "intermediate wrappers by JFM (users: instead use finufftpy.*)";

  // 1-d
  m.def("finufft1d1_cpp", &finufft1d1_cpp, "Python wrapper for 1-d type 1 nufft",
        py::arg("xj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),py::arg("ms"),
	py::arg("fk").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("modeord"),py::arg("chkbnds"),py::arg("upsampfac"));
  
  m.def("finufft1d2_cpp", &finufft1d2_cpp, "Python wrapper for 1-d type 2 nufft",
        py::arg("xj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),
	py::arg("fk").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("modeord"),py::arg("chkbnds"),py::arg("upsampfac"));

    
  m.def("finufft1d3_cpp", &finufft1d3_cpp, "Python wrapper for 1-d type 3 nufft",
        py::arg("x").noconvert(),py::arg("c").noconvert(),
        py::arg("iflag"),py::arg("eps"),
	py::arg("s").noconvert(),py::arg("f").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("upsampfac"));
  
  
  // 2-d
  m.def("finufft2d1_cpp", &finufft2d1_cpp, "Python wrapper for 2-d type 1 nufft",
        py::arg("xj").noconvert(),py::arg("yj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),py::arg("ms"),py::arg("mt"),
	py::arg("fk").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("modeord"),py::arg("chkbnds"),py::arg("upsampfac"));
  
  m.def("finufft2d1many_cpp", &finufft2d1many_cpp, "Python wrapper for 2-d type 1 many nufft",
        py::arg("xj").noconvert(),py::arg("yj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),py::arg("ms"),py::arg("mt"),
	py::arg("fk").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("modeord"),py::arg("chkbnds"),py::arg("upsampfac"));
  
  
  m.def("finufft2d2_cpp", &finufft2d2_cpp, "Python wrapper for 2-d type 2 nufft",
        py::arg("xj").noconvert(),py::arg("yj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),
	py::arg("fk").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("modeord"),py::arg("chkbnds"),py::arg("upsampfac"));
  
  m.def("finufft2d2many_cpp", &finufft2d2many_cpp, "Python wrapper for 2-d type 2 many nufft",
        py::arg("xj").noconvert(),py::arg("yj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),
	py::arg("fk").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("modeord"),py::arg("chkbnds"),py::arg("upsampfac"));
 
  m.def("finufft2d3_cpp", &finufft2d3_cpp, "Python wrapper for 2-d type 3 nufft",
        py::arg("x").noconvert(),py::arg("y").noconvert(),py::arg("c").noconvert(),
        py::arg("iflag"),py::arg("eps"),
	py::arg("s").noconvert(),py::arg("t").noconvert(),py::arg("f").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("upsampfac"));
  
  
  // 3-d
  m.def("finufft3d1_cpp", &finufft3d1_cpp, "Python wrapper for 3-d type 1 nufft",
        py::arg("xj").noconvert(),py::arg("yj").noconvert(),py::arg("zj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),py::arg("ms"),py::arg("mt"),py::arg("mu"),
	py::arg("fk").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("modeord"),py::arg("chkbnds"),py::arg("upsampfac"));
  
  
  m.def("finufft3d2_cpp", &finufft3d2_cpp, "Python wrapper for 3-d type 2 nufft",
        py::arg("xj").noconvert(),py::arg("yj").noconvert(),py::arg("zj").noconvert(),py::arg("cj").noconvert(),
        py::arg("iflag"),py::arg("eps"),
	py::arg("fk").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("modeord"),py::arg("chkbnds"),py::arg("upsampfac"));
  
  
  m.def("finufft3d3_cpp", &finufft3d3_cpp, "Python wrapper for 3-d type 3 nufft",
        py::arg("x").noconvert(),py::arg("y").noconvert(),py::arg("z").noconvert(),py::arg("c").noconvert(),
        py::arg("iflag"),py::arg("eps"),
	py::arg("s").noconvert(),py::arg("t").noconvert(),py::arg("u").noconvert(),py::arg("f").noconvert(),
	py::arg("debug"),py::arg("spread_debug"),py::arg("spread_sort"),
	py::arg("fftw"),py::arg("upsampfac"));

}   // end of module
*/
