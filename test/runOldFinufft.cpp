#include <finufft.h>
#include <finufft_old.h>
#include <utils.h>

//forward declaration
double finufftFunnel(CPX *cStart, CPX *fStart, finufft_plan *plan);

/*A unified interface to all of the old implementations*/ 
double runOldFinufft(CPX *c,CPX *F,finufft_plan *plan){
    
    CPX *cStart;
    CPX *fStart;

    double time = 0;
    double temp = 0;;
    int ier = 0;
    
    for(int k = 0; k < plan->n_transf; k++){
      cStart = c + plan->nj*k;
      fStart = F + plan->ms*plan->mt*plan->mu*k;
      
      if(k != 0){
	plan->opts.debug = 0;
	plan->opts.spread_debug = 0;
      }
      
      temp = finufftFunnel(cStart,fStart, plan);
      if(temp == -1){
	printf("Call to finufft FAILED!"); 
	time = -1;
	break;
      }
      else
	time += temp;
    }
    return time;
}



double finufftFunnel(CPX *cStart, CPX *fStart, finufft_plan *plan){

  CNTime timer; timer.start();
  int ier = 0;
  double t = 0;
  switch(plan->n_dims){

    /*1d*/
  case 1:
    switch(plan->type){

    case type1:
      timer.restart();
      ier = finufft1d1_old(plan->nj, plan->X, cStart, plan->iflag, plan->tol, plan->ms, fStart, plan->opts);
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;
      
    case type2:
      timer.restart();
      ier = finufft1d2_old(plan->nj, plan->X, cStart, plan->iflag, plan->tol, plan->ms, fStart, plan->opts);
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;
      
    case type3:
      timer.restart();
      ier = finufft1d3_old(plan->nj, plan->X, cStart, plan->iflag, plan->tol, plan->nk, plan->s, fStart, plan->opts);
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;
      
    default:
      return -1; 

    }

    /*2d*/
  case 2:
    switch(plan->type){
      
    case type1:
      timer.restart();
      ier = finufft2d1_old(plan->nj, plan->X, plan->Y, cStart, plan->iflag, plan->tol, plan->ms, plan->mt,
     		       fStart, plan->opts);
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;
      
    case type2:
      timer.restart();
      ier = finufft2d2_old(plan->nj, plan->X, plan->Y, cStart, plan->iflag, plan->tol, plan->ms, plan->mt,
     		       fStart, plan->opts);
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;

    case type3:
      timer.restart();
      ier = finufft2d3_old(plan->nj, plan->X, plan->Y, cStart, plan->iflag, plan->tol, plan->nk, plan->s, plan->t,
			   fStart, plan->opts); 
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;
      
    default:
      return -1;
    }

    /*3d*/
  case 3:
    
    switch(plan->type){

    case type1:
      timer.restart();
      ier = finufft3d1_old(plan->nj, plan->X, plan->Y, plan->Z, cStart, plan->iflag, plan->tol,
			   plan->ms, plan->mt, plan->mu, fStart, plan->opts);
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;
      
    case type2:
      timer.restart();
      ier = finufft3d2_old(plan->nj, plan->X, plan->Y, plan->Z, cStart, plan->iflag, plan->tol,
			   plan->ms, plan->mt, plan->mu, fStart, plan->opts);
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;
      
    case type3:
      timer.restart();
      ier = finufft3d3_old(plan->nj, plan->X, plan->Y, plan->Z, cStart, plan->iflag, plan->tol,
			   plan->nk, plan->s, plan->t, plan->u, fStart, plan->opts);
      t = timer.elapsedsec();
      if(ier)
	return -1;
      else
	return t;

    /*invalid type*/
    default:
      return -1;
    }

    /*invalid dimension*/
  default:
    return -1;
  }
}




