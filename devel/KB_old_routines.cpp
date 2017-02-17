

#include "../contrib/besseli.h"


// obsolete KB stuff from .h
double evaluate_KB_kernel(double x,const spread_opts &opts);
int set_KB_opts_from_kernel_params(spread_opts &opts,double *kernel_params);
int set_KB_opts_from_eps(spread_opts &opts,double eps);
int get_KB_kernel_params_for_eps(double *kernel_params,double eps);



struct spread_opts {
  // KB kernel specific... (obsolete)
  double KB_fac1=1;
  double KB_fac2=1;
  void set_KB_W_and_beta();   // must be called before using KB kernel
  double KB_W;             // derived params, only experts should change...
  double KB_beta;
};



//********************** obsolete Kaiser-Bessel kernel routines *************

double evaluate_KB_kernel(double x, const spread_opts &opts)
{
  //if (abs(x)>2.0) return 0.0; else return exppolyker(x,a4,6,2);  // expt ns=4 (1e-2)***
  //return 0.0; //exp(x); // to test how much time spent on kernel eval
  // todo: insert test if opts.kernel_type==1 ?
  double t = 2.0*x/opts.KB_W;
  //printf("x=%g\n",x);
  double tmp1=1.0-t*t;
  if (tmp1<0.0) {
     return 0.0;      // you fell outside the support
  } else {
    double y = opts.KB_beta*sqrt(tmp1);  // set up arg for I_0
    //return besseli0(y);  // full-acc version, does slow whole thing down
    return besseli0_approx(y);  // the meat, lower-acc version has little effect
  }
}

void spread_opts::set_KB_W_and_beta() {  // set derived parameters in Kaiser--Bessel
  this->KB_W = this->nspread * this->KB_fac1;
  double tmp0 = this->KB_W * this->KB_W / 4 - 0.8;
  if (tmp0<0) tmp0=0;   // fix it?
  this->KB_beta = M_PI*sqrt(tmp0) * this->KB_fac2;
}

int set_KB_opts_from_kernel_params(spread_opts &opts,double *kernel_params) {
/* Specific to Kaiser-Bessel kernel. Directly sets kernel options from a param array.
 * Returns 0 is success, or 1 error code if params[0] is not KB type.
 *
 * kernel_params is a 4-element double array containing following information:
 *  entry 0: kernel type (1 for Kaiser--Bessel; others report an error)
 *  entry 1: nspread
 *  entry 2: KB_fac1 (eg 1)
 *  entry 3: KB_fac2 (eg 1)
 */
  if (kernel_params[0]!=1) {
    fprintf(stderr,"error: unsupported kernel type param[0]=%g\n",kernel_params[0]);
    return 1;
  }
  opts.nspread=kernel_params[1];
  opts.KB_fac1=kernel_params[2];
  opts.KB_fac2=kernel_params[3];
  opts.set_KB_W_and_beta();
  return 0;
}

int set_KB_opts_from_eps(spread_opts &opts,double eps)
// Specific to KB kernel. Sets spreading opts from accuracy eps.
// nspread should always be even, based on other uses in this file.
// Returns 0 if success, or 1 error code if eps out of range.
{
  int nspread=12; double fac1=1,fac2=1;  // defaults: todo decide for what tol?
  // tests done sequentially to categorize eps...
  if (eps>=1e-1) {
    nspread=2; fac1=1.0; fac2=2.0;   // ahb guess
  } else if (eps>=1e-2) {
    nspread=4; fac1=0.75; fac2=1.71;
  } else if (eps>=1e-4) {
    nspread=6; fac1=0.83; fac2=1.56;
  } else if (eps>=1e-6) {
    nspread=8; fac1=0.89; fac2=1.45;
  } else if (eps>=1e-8) {
    nspread=10; fac1=0.90; fac2=1.47;
  } else if (eps>=1e-10) {
    nspread=12; fac1=0.92; fac2=1.51;
  } else if (eps>=1e-12) {
    nspread=14; fac1=0.94; fac2=1.48;
  } else {
    nspread=16; fac1=0.94; fac2=1.46;
  }
  opts.nspread=nspread;
  opts.KB_fac1=fac1;
  opts.KB_fac2=fac2;
  opts.set_KB_W_and_beta();
  if (eps<1e-16) {      // report problem but don't exit
    fprintf(stderr,"set_kb_opts_from_eps: eps too small!\n");
    return 1;
  }
  return 0;
}

int get_KB_kernel_params_for_eps(double *kernel_params,double eps)
{
  spread_opts opts;
  int ier = set_KB_opts_from_eps(opts,eps);
  kernel_params[0]=1;
  kernel_params[1]=opts.nspread;
  kernel_params[2]=opts.KB_fac1;
  kernel_params[3]=opts.KB_fac2;
  return ier;
}

// *************************** end obsolete KB routines ***********************
