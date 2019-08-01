// ------------------ Deprecated Interface ------------------------------------

void finufft_default_opts(nufft_opts *o);

int finufft1d1(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts opts);

int finufft1d2(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
	       CPX* fk, nufft_opts opts);

int finufft1d3(BIGINT nj,FLT* x,CPX* c,int iflag,FLT eps,BIGINT nk, FLT* s, CPX* f, nufft_opts opts);


int finufft2d1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);


int finufft2d2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);

int finufft2d2many(int ntransf, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts); //Melody

int finufft2d3(BIGINT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,BIGINT nk, FLT* s, FLT* t, CPX* fk, nufft_opts opts);

  
int finufft3d1(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);


int finufft3d2(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);


int finufft3d3(BIGINT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
	       FLT eps,BIGINT nk,FLT* s, FLT* t, FLT *u,
	       CPX* fk, nufft_opts opts);

/*********The Problem*************************************/
//Originally 
for(int i = 0; i < n_transforms; i++){
  finufft?d?();
 }

//translates to
for(int i = 0; i < n_transforms; i++){
  fftw_plan plan;
  fftw_exec(plan);
 }


// ------------------Halfway Solution--------------------------------------
int finufft1d1many(int ntransf, BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
		   CPX* fk, nufft_opts opts); //new addition


int finufft1d2many(int ntransf, BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms, 
		   CPX* fk, nufft_opts opts); //new addition

int finufft1d3many(int ntransf, BIGINT nj,FLT* x,CPX* c,int iflag,FLT eps,BIGINT nk, FLT* s, CPX* f, nufft_opts opts);  //new addition


int finufft2d1many(int ntransf, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts); //new addition

int finufft2d3many(int ntransf, BIGINT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,BIGINT nk, FLT* s, FLT* t, CPX* fk, nufft_opts opts); //new addition

int finufft3d1many(int ntransfs, BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);

int finufft3d2many(int ntransf, BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
		   BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts); //new addition

int finufft3d3many(int ntransf, BIGINT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
	       FLT eps,BIGINT nk,FLT* s, FLT* t, FLT *u,
		   CPX* fk, nufft_opts opts); //new addition 


/**********************************************/
//Intermediary [fftw plan time overhead reduction achieved]
finufft?d?many();

//translates to
{
  fftw_plan plan;
  for(int i = 0; i  < n_transforms; i++)
    fftw_exec(plan);
}

//BUT!
[3dimension * 3types * {single,many}] = 18 routines

//AUDIENCE QUIZ
  
// ------------------ Wholeway Solution ------ Guru Interface ------------------------------------

int finufft_makeplan(finufft_type type, int n_dims, BIGINT* n_modes, int iflag, int n_transf, FLT tol, int blksize, finufft_plan *plan );
void finufft_default_opts(nufft_opts *o);
int finufft_setpts(finufft_plan * plan , BIGINT M, FLT *xj, FLT *yj, FLT *zj, BIGINT N, FLT *s, FLT *t, FLT *u); 
int finufft_exec(finufft_plan * plan ,  CPX *weights, CPX * result);
int finufft_destroy(finufft_plan * plan);

/**********************************************/
//newest regime [source code reduction]
{
  finufft_makeplan();
  finufft_default_opts();
  finufft_setpts();
  finufft_exec();
  finufft_destroy();
}







  
