
/*Original*/
finufft1d1(){
  .
  .
  .
  .
  .
  .
  timer.restart();
  spopts.spread_direction = 1;
  FLT *dummy=NULL;
  int ier_spread = spread(nf1,1,1,(FLT*)fw,nj,xj,dummy,dummy,(FLT*)cj,spopts);
  .
  .
  .
  .
  .
  .
  .
  for (BIGINT k=0;k<nk;++k)
    sp[k] = h1*gam1*(s[k]-D1);                         // so that |s'_k| < pi/R
  .
  .
  .
  .
  .
}

/* New */
finufft_exec(){
  .
  .
  .
  .
  .
  .
  .
  /*Five extra multiplies*/
  BIGINT fwRowSize = plan->nf1*plan->nf2*plan->nf3; 
  int blkJump = blkNum*plan->threadBlkSize; 

  for(int i = 0; i < transfThisRound; i++){ 

    //index into this iteration of fft in fw and weights arrays
    FFTW_CPX *fwStart = plan->fw + fwRowSize*i;

    CPX *cStart;
    if(plan->type == type3)
      cStart = c + plan->nj*i;

    else
      cStart = c + plan->nj*(i + blkJump); 
    
    int ier = spread(plan->sortIndices,
                     plan->nf1, plan->nf2, plan->nf3, (FLT*)fwStart,
                     plan->nj, plan->X, plan->Y, plan->Z, (FLT *)cStart,
                     plan->spopts, plan->didSort) ;
  }

  .
  .
  .
  .
  .
  .
  .

    /*Worst case, triple the work?*/
#pragma omp parallel for schedule(static) //static appropriate for load balance across loop iterations 
    for (BIGINT k=0;k<plan->nk;++k) {
	sp[k] = plan->t3P.h1*plan->t3P.gam1*(s[k]-plan->t3P.D1);     
	if(plan->n_dims > 1 )
	  tp[k] = plan->t3P.h2*plan->t3P.gam2*(t[k]-plan->t3P.D2);  
	if(plan->n_dims > 2)
	  up[k] = plan->t3P.h3*plan->t3P.gam3*(u[k]-plan->t3P.D3);  
    }
  .
  .
  .
  .
  .
  .
  .    
}
