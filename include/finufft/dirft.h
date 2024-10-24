#ifndef DIRFT_H
#define DIRFT_H

#include <finufft/finufft_core.h>

template<typename T>
void dirft1d1(BIGINT nj, T *x, std::complex<T> *c, int isign, BIGINT ms,
              std::complex<T> *f);
template<typename T>
void dirft1d2(BIGINT nj, T *x, std::complex<T> *c, int iflag, BIGINT ms,
              std::complex<T> *f);
template<typename T>
void dirft1d3(BIGINT nj, T *x, std::complex<T> *c, int iflag, BIGINT nk, T *s,
              std::complex<T> *f);

template<typename T>
void dirft2d1(BIGINT nj, T *x, T *y, std::complex<T> *c, int iflag, BIGINT ms, BIGINT mt,
              std::complex<T> *f);
template<typename T>
void dirft2d2(BIGINT nj, T *x, T *y, std::complex<T> *c, int iflag, BIGINT ms, BIGINT mt,
              std::complex<T> *f);
template<typename T>
void dirft2d3(BIGINT nj, T *x, T *y, std::complex<T> *c, int iflag, BIGINT nk, T *s, T *t,
              std::complex<T> *f);

template<typename T>
void dirft3d1(BIGINT nj, T *x, T *y, T *z, std::complex<T> *c, int iflag, BIGINT ms,
              BIGINT mt, BIGINT mu, std::complex<T> *f);
template<typename T>
void dirft3d2(BIGINT nj, T *x, T *y, T *z, std::complex<T> *c, int iflag, BIGINT ms,
              BIGINT mt, BIGINT mu, std::complex<T> *f);
template<typename T>
void dirft3d3(BIGINT nj, T *x, T *y, T *z, std::complex<T> *c, int iflag, BIGINT nk, T *s,
              T *t, T *u, std::complex<T> *f);

#endif
