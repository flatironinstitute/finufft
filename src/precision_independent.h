/* These are functions that do not rely on FLT.
   They are organized by originating file.
*/

#ifndef PRECISION_INDEPENDENT_H
#define PRECISION_INDEPENDENT_H

/* Common Kernels from spreadinterp3d */
__device__
int CalcGlobalIdx(int xidx, int yidx, int zidx, int onx, int ony, int onz,
                  int bnx, int bny, int bnz);
__device__
int CalcGlobalIdx_V2(int xidx, int yidx, int zidx, int nbinx, int nbiny, int nbinz);

/* spreadinterp 2d */
__global__
void CalcSubProb_2d(int* bin_size, int* num_subprob, int maxsubprobsize, int numbins);

__global__
void MapBintoSubProb_2d(int* d_subprob_to_bin, int* d_subprobstartpts,
	int* d_numsubprob,int numbins);

__global__
void CalcSubProb_2d_Paul(int* finegridsize, int* num_subprob,
	int maxsubprobsize, int bin_size_x, int bin_size_y);

__global__
void TrivialGlobalSortIdx_2d(int M, int* index);

/* spreadinterp3d */
__global__
void CalcSubProb_3d_v2(int* bin_size, int* num_subprob, int maxsubprobsize,
                       int numbins);

__global__
void MapBintoSubProb_3d_v2(int* d_subprob_to_bin,int* d_subprobstartpts,
                           int* d_numsubprob,int numbins);

__global__
void CalcSubProb_3d_v1(int binsperobinx, int binsperobiny, int binsperobinz,
                       int* bin_size, int* num_subprob, int maxsubprobsize, int numbins);

__global__
void MapBintoSubProb_3d_v1(int* d_subprob_to_obin, int* d_subprobstartpts,
                           int* d_numsubprob,int numbins);

__global__
void TrivialGlobalSortIdx_3d(int M, int* index);

__global__
void FillGhostBins(int binsperobinx, int binsperobiny, int binsperobinz,
                   int nobinx, int nobiny, int nobinz, int* binsize);

__global__
void Temp(int binsperobinx, int binsperobiny, int binsperobinz,
          int nobinx, int nobiny, int nobinz, int* binsize);

__global__
void GhostBinPtsIdx(int binsperobinx, int binsperobiny, int binsperobinz,
                    int nobinx, int nobiny, int nobinz, int* binsize, int* index,
                    int* binstartpts, int M);

#endif
