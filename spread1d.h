#ifndef __SPREAD1D_H__
#define __SPREAD1D_H__

__device__
double evaluate_kernel(double x, double es_c, double es_beta);

__global__
void CalcBinSize(int M, int nf1, int  bin_size_x, unsigned int* bin_size, double *x, unsigned int* sortidx);

__global__
void BinsStartPts(int M, int numbins, unsigned int* bin_size, unsigned int* bin_startpts);

__global__
void PtsRearrage(int M, int nf1, int bin_size_x, int numbins, unsigned int* bin_startpts, unsigned int* sortidx,
                 double* x, double* x_sorted,
                 cuDoubleComplex* c, cuDoubleComplex* c_sorted);

__global__
void Spread(unsigned int numbinperblock, unsigned int* bin_startpts, double* x_sorted,
            cuDoubleComplex* c_sorted, cuDoubleComplex* fw, int ns, int nf1, double es_c,
            double es_beta);
#endif
