#include <finufft_plan.h>
#include <fftw_util.h>

int* GRIDSIZE_FOR_FFTW(finufft_plan* p){
// helper func returns a new int array of length dim, extracted from
// the finufft plan, that fftw_plan_many_dft needs as its 2nd argument.
  int* nf;
  if(p->dim == 1){ 
    nf = new int[1];
    nf[0] = (int)p->nf1;
  }
  else if (p->dim == 2){ 
    nf = new int[2];
    nf[0] = (int)p->nf2;
    nf[1] = (int)p->nf1; 
  }   // fftw enforced row major ordering, ie dims are backwards ordered
  else{ 
    nf = new int[3];
    nf[0] = (int)p->nf3;
    nf[1] = (int)p->nf2;
    nf[2] = (int)p->nf1;
  }
  return nf;
}
