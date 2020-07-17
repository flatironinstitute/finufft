/* These are functions that do not rely on FLT.
   They are organized by originating file.
*/

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "precision_independent.h"

/* Common Kernels from spreadinterp3d */
__device__
int CalcGlobalIdx(int xidx, int yidx, int zidx, int onx, int ony, int onz,
	int bnx, int bny, int bnz){
	int oix,oiy,oiz;
	oix = xidx/bnx;
	oiy = yidx/bny;
	oiz = zidx/bnz;
	return   (oix + oiy*onx + oiz*ony*onx)*(bnx*bny*bnz) +
			 (xidx%bnx+yidx%bny*bnx+zidx%bnz*bny*bnx);
}

__device__
int CalcGlobalIdx_V2(int xidx, int yidx, int zidx, int nbinx, int nbiny, int nbinz){
	return xidx + yidx*nbinx + zidx*nbinx*nbiny;
}

/* spreadinterp 2d */
__global__
void CalcSubProb_2d(int* bin_size, int* num_subprob, int maxsubprobsize,
	int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins;
		i+=gridDim.x*blockDim.x){
		num_subprob[i]=ceil(bin_size[i]/(float) maxsubprobsize);
	}
}

__global__
void MapBintoSubProb_2d(int* d_subprob_to_bin,int* d_subprobstartpts,
	int* d_numsubprob,int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins;
		i+=gridDim.x*blockDim.x){
		for(int j=0; j<d_numsubprob[i]; j++){
			d_subprob_to_bin[d_subprobstartpts[i]+j]=i;
		}
	}
}

__global__
void CalcSubProb_2d_Paul(int* finegridsize, int* num_subprob,
	int maxsubprobsize, int bin_size_x, int bin_size_y)
{
	int binsize = bin_size_x*bin_size_y;
	int *maxptsinbin = thrust::max_element(thrust::seq,
			finegridsize+binsize*blockIdx.x,
			finegridsize + binsize*(blockIdx.x+1));
	num_subprob[blockIdx.x] = (int)ceil(*maxptsinbin/(float) maxsubprobsize);
}

__global__
void TrivialGlobalSortIdx_2d(int M, int* index)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		index[i] = i;
	}
}

/* spreadinterp3d */
__global__
void CalcSubProb_3d_v2(int* bin_size, int* num_subprob, int maxsubprobsize,
	int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins;
		i+=gridDim.x*blockDim.x){
		num_subprob[i]=ceil(bin_size[i]/(float) maxsubprobsize);
	}
}

__global__
void MapBintoSubProb_3d_v2(int* d_subprob_to_bin,int* d_subprobstartpts,
	int* d_numsubprob,int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins;
		i+=gridDim.x*blockDim.x){
		for(int j=0; j<d_numsubprob[i]; j++){
			d_subprob_to_bin[d_subprobstartpts[i]+j]=i;
		}
	}
}

__global__
void CalcSubProb_3d_v1(int binsperobinx, int binsperobiny, int binsperobinz,
	int* bin_size, int* num_subprob, int maxsubprobsize, int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins; i+=gridDim.x*
		blockDim.x){
		int numnupts = 0;
		int binsperobin = binsperobinx*binsperobiny*binsperobinz;
		for(int b = 0; b<binsperobin; b++){
			numnupts += bin_size[binsperobin*i+b];
		}
		num_subprob[i]=ceil(numnupts/(float) maxsubprobsize);
	}
}

__global__
void MapBintoSubProb_3d_v1(int* d_subprob_to_obin, int* d_subprobstartpts,
	int* d_numsubprob,int numbins)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<numbins;
		i+=gridDim.x*blockDim.x){
		for(int j=0; j<d_numsubprob[i]; j++){
			d_subprob_to_obin[d_subprobstartpts[i]+j]=i;
		}
	}
}

__global__
void TrivialGlobalSortIdx_3d(int M, int* index)
{
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		index[i] = i;
	}
}

__global__
void FillGhostBins(int binsperobinx, int binsperobiny, int binsperobinz,
	int nobinx, int nobiny, int nobinz, int* binsize)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;

	int nbinx = nobinx*binsperobinx;
	int nbiny = nobiny*binsperobiny;
	int nbinz = nobinz*binsperobinz;

	if(binx < nbinx && biny < nbiny && binz < nbinz){
		int binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,
			binsperobinx,binsperobiny, binsperobinz);
		int i,j,k;
		i = binx;
		j = biny;
		k = binz;
		if(binx % binsperobinx == 0){
			i = binx - 2;
			i = i<0 ? i+nbinx : i;
		}
		if(binx % binsperobinx == binsperobinx-1){
			i = binx + 2;
			i = (i>=nbinx) ? i-nbinx : i;
		}
		if(biny % binsperobiny == 0){
			j = biny - 2;
			j = j<0 ? j+nbiny : j;
		}
		if(biny % binsperobiny == binsperobiny-1){
			j = biny + 2;
			j = (j>=nbiny) ? j-nbiny : j;
		}
		if(binz % binsperobinz == 0){
			k = binz - 2;
			k = k<0 ? k+nbinz : k;
		}
		if(binz % binsperobinz == binsperobinz-1){
			k = binz + 2;
			k = (k>=nbinz) ? k-nbinz : k;
		}
		int idxtoupdate = CalcGlobalIdx(i,j,k,nobinx,nobiny,nobinz,
				binsperobinx,binsperobiny, binsperobinz);
		if(idxtoupdate != binidx){
			binsize[binidx] = binsize[idxtoupdate];
		}
	}
}

__global__
void Temp(int binsperobinx, int binsperobiny, int binsperobinz,
	int nobinx, int nobiny, int nobinz, int* binsize)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;
	int binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,binsperobinx,
			binsperobiny, binsperobinz);

	if(binx < nobinx*binsperobinx && biny < nobiny*binsperobiny &&
		binz < nobinz*binsperobinz)
		if (binx%binsperobinx >0 && binx%binsperobinx< binsperobinx-1)
			if (biny%binsperobiny >0 && biny%binsperobiny< binsperobiny-1)
				if (binz%binsperobinz >0 && binz%binsperobinz< binsperobinz-1)
					binsize[binidx] = binidx;
}

__global__
void GhostBinPtsIdx(int binsperobinx, int binsperobiny, int binsperobinz,
	int nobinx, int nobiny, int nobinz, int* binsize, int* index,
	int* binstartpts, int M)
{
	int binx =threadIdx.x+blockIdx.x*blockDim.x;
	int biny =threadIdx.y+blockIdx.y*blockDim.y;
	int binz =threadIdx.z+blockIdx.z*blockDim.z;
	int nbinx = nobinx*binsperobinx;
	int nbiny = nobiny*binsperobiny;
	int nbinz = nobinz*binsperobinz;

	int i,j,k;
	int w = 0;
	int box[3];
	if(binx < nbinx && biny < nbiny && binz < nbinz){
		box[0] = box[1] = box[2] = 0;
		i = binx;
		j = biny;
		k = binz;
		int binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,
			binsperobinx,binsperobiny,binsperobinz);
		if(binx % binsperobinx == 0){
			i = binx - 2;
			box[0] = (i<0);
			i = i<0 ? i+nbinx : i;
			w=1;
		}
		if(binx % binsperobinx == binsperobinx-1){
			i = binx + 2;
			box[0] = (i>nbinx)*2;
			i = (i>nbinx) ? i-nbinx : i;
			w=1;
		}
		if(biny % binsperobiny == 0){
			j = biny - 2;
			box[1] = (j<0);
			j = j<0 ? j+nbiny : j;
			w=1;
		}
		if(biny % binsperobiny == binsperobiny-1){
			j = biny + 2;
			box[1] = (j>nbiny)*2;
			j = (j>nbiny) ? j-nbiny : j;
			w=1;
		}
		if(binz % binsperobinz == 0){
			k = binz - 2;
			box[2] = (k<0);
			k = k<0 ? k+nbinz : k;
			w=1;
		}
		if(binz % binsperobinz == binsperobinz-1){
			k = binz + 2;
			box[2] = (k>nbinz)*2;
			k = (k>nbinz) ? k-nbinz : k;
			w=1;
		}
		int corbinidx = CalcGlobalIdx(i,j,k,nobinx,nobiny,nobinz,
			binsperobinx,binsperobiny, binsperobinz);
		if(w==1){
			for(int n = 0; n<binsize[binidx];n++){
				index[binstartpts[binidx]+n] = M*(box[0]+box[1]*3+box[2]*9) +
					index[binstartpts[corbinidx]+n];
			}
		}
	}
}
