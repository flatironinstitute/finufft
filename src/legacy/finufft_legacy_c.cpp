#include <finufft_legacy.h>
#include <finufft_legacy_c.h>
#include <nufft_opts.h>

nufft_opts *create_opts(){
  return new nufft_opts();
}
void destroy_opts(nufft_opts *opts){
  delete opts;
}

void set_opts_debug(nufft_opts *opts, int debug){
  opts->debug = debug;
}

void finufft_default_opts_c(nufft_opts *o){
  finufft_default_opts(o);
}

int finufft1d1_c(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts *opts){
  return finufft1d1(nj,xj,cj,iflag,eps,ms,fk,*opts);
}
