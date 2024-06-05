#include <finufft/defs.h>
#include <finufft/dirft.h>
#include <iostream>

// This is basically a port of dirft2d.f from CMCL package, except with
// the 1/nj prefactors for type-1 removed.

void dirft2d1(BIGINT nj, FLT *x, FLT *y, CPX *c, int iflag, BIGINT ms, BIGINT mt, CPX *f)
/* Direct computation of 2D type-1 nonuniform FFT. Interface same as finufft2d1.
c                  nj-1
c     f[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
c                  j=0
c
c     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.
c     The output array is in increasing k1 ordering (fast), then increasing
      k2 ordering (slow). If iflag>0 the + sign is
c     used, otherwise the - sign is used, in the exponential.
*  Uses C++ complex type and winding trick.  Barnett 1/26/17
*/
{
  BIGINT k1min = -(ms / 2), k2min = -(mt / 2);     // integer divide
  BIGINT N = ms * mt;                              // total # output modes
  for (BIGINT m = 0; m < N; ++m) f[m] = CPX(0, 0); // it knows f is complex type
  for (BIGINT j = 0; j < nj; ++j) {                // src pts
    CPX a1   = (iflag > 0) ? exp(IMA * x[j]) : exp(-IMA * x[j]);
    CPX a2   = (iflag > 0) ? exp(IMA * y[j]) : exp(-IMA * y[j]);
    CPX sp1  = pow(a1, (FLT)k1min);        // starting phase for most neg k1 freq
    CPX p2   = pow(a2, (FLT)k2min);
    CPX cc   = c[j];                       // no 1/nj norm
    BIGINT m = 0;                          // output pointer
    for (BIGINT m2 = 0; m2 < mt; ++m2) {
      CPX p1 = sp1;                        // must reset p1 for each inner loop
      for (BIGINT m1 = 0; m1 < ms; ++m1) { // ms is fast, mt slow
        f[m++] += cc * p1 * p2;
        p1 *= a1;
      }
      p2 *= a2;
    }
  }
}

void dirft2d2(BIGINT nj, FLT *x, FLT *y, CPX *c, int iflag, BIGINT ms, BIGINT mt, CPX *f)
/* Direct computation of 2D type-2 nonuniform FFT. Interface same as finufft2d2

     c[j] = SUM   f[k1,k2] exp(+-i (k1 x[j] + k2 y[j]))
            k1,k2
                            for j = 0,...,nj-1
    where sum is over -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

    The input array is in increasing k1 ordering (fast), then increasing
    k2 ordering (slow).
    If iflag>0 the + sign is used, otherwise the - sign is used, in the
    exponential.
    Uses C++ complex type and winding trick.  Barnett 1/26/17
*/
{
  BIGINT k1min = -(ms / 2), k2min = -(mt / 2); // integer divide
  for (BIGINT j = 0; j < nj; ++j) {
    CPX a1   = (iflag > 0) ? exp(IMA * x[j]) : exp(-IMA * x[j]);
    CPX a2   = (iflag > 0) ? exp(IMA * y[j]) : exp(-IMA * y[j]);
    CPX sp1  = pow(a1, (FLT)k1min);
    CPX p2   = pow(a2, (FLT)k2min);
    CPX cc   = CPX(0, 0);
    BIGINT m = 0; // input pointer
    for (BIGINT m2 = 0; m2 < mt; ++m2) {
      CPX p1 = sp1;
      for (BIGINT m1 = 0; m1 < ms; ++m1) {
        cc += f[m++] * p1 * p2;
        p1 *= a1;
      }
      p2 *= a2;
    }
    c[j] = cc;
  }
}

void dirft2d3(BIGINT nj, FLT *x, FLT *y, CPX *c, int iflag, BIGINT nk, FLT *s, FLT *t,
              CPX *f)
/* Direct computation of 2D type-3 nonuniform FFT. Interface same as finufft2d3
c               nj-1
c     f[k]  =   SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j]))
c               j=0
c                    for k = 0, ..., nk-1
c  If iflag>0 the + sign is used, otherwise the - sign is used, in the
c  exponential. Uses C++ complex type. Simple brute force.  Barnett 1/26/17
*/
{
  for (BIGINT k = 0; k < nk; ++k) {
    CPX ss = (iflag > 0) ? IMA * s[k] : -IMA * s[k];
    CPX tt = (iflag > 0) ? IMA * t[k] : -IMA * t[k];
    f[k]   = CPX(0, 0);
    for (BIGINT j = 0; j < nj; ++j) f[k] += c[j] * exp(ss * x[j] + tt * y[j]);
  }
}
