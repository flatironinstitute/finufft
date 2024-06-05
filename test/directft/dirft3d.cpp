#include <finufft/defs.h>
#include <finufft/dirft.h>
#include <iostream>

// This is basically a port of dirft2d.f from CMCL package, except with
// the 1/nj prefactors for type-1 removed.

void dirft3d1(BIGINT nj, FLT *x, FLT *y, FLT *z, CPX *c, int iflag, BIGINT ms, BIGINT mt,
              BIGINT mu, CPX *f)
/* Direct computation of 3D type-1 nonuniform FFT. Interface same as finufft3d1.
c                     nj-1
c     f[k1,k2,k3] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k2 z[j]))
c                     j=0
c
c     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
          -mu/2 <= k3 <= (mu-1)/2,
c     The output array is in increasing k1 ordering (fast), then increasing
      k2 ordering (medium), then increasing k3 (fast). If iflag>0 the + sign is
c     used, otherwise the - sign is used, in the exponential.
*  Uses C++ complex type and winding trick.  Barnett 2/1/17
*/
{
  BIGINT k1min = -(ms / 2), k2min = -(mt / 2), k3min = -(mu / 2); // integer divide
  BIGINT N = ms * mt * mu;                                        // total # output modes
  for (BIGINT m = 0; m < N; ++m) f[m] = CPX(0, 0); // it knows f is complex type
  for (BIGINT j = 0; j < nj; ++j) {                // src pts
    CPX a1   = (iflag > 0) ? exp(IMA * x[j]) : exp(-IMA * x[j]);
    CPX a2   = (iflag > 0) ? exp(IMA * y[j]) : exp(-IMA * y[j]);
    CPX a3   = (iflag > 0) ? exp(IMA * z[j]) : exp(-IMA * z[j]);
    CPX sp1  = pow(a1, (FLT)k1min); // starting phase for most neg k1 freq
    CPX sp2  = pow(a2, (FLT)k2min);
    CPX p3   = pow(a3, (FLT)k3min);
    CPX cc   = c[j]; // no 1/nj norm
    BIGINT m = 0;    // output pointer
    for (BIGINT m3 = 0; m3 < mu; ++m3) {
      CPX p2 = sp2;
      for (BIGINT m2 = 0; m2 < mt; ++m2) {
        CPX p1 = sp1;
        for (BIGINT m1 = 0; m1 < ms; ++m1) {
          f[m++] += cc * p1 * p2 * p3;
          p1 *= a1;
        }
        p2 *= a2;
      }
      p3 *= a3;
    }
  }
}

void dirft3d2(BIGINT nj, FLT *x, FLT *y, FLT *z, CPX *c, int iflag, BIGINT ms, BIGINT mt,
              BIGINT mu, CPX *f)
/* Direct computation of 3D type-2 nonuniform FFT. Interface same as finufft3d2

     c[j] =   SUM    f[k1,k2,k3] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
            k1,k2,k3
                            for j = 0,...,nj-1
    where sum is over -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
                      -mu/2 <= k3 <= (mu-1)/2.

    The input array is in increasing k1 ordering (fast), then increasing
    k2 ordering (medium), then increasing k3 (fast).
    If iflag>0 the + sign is used, otherwise the - sign is used, in the
    exponential.
    Uses C++ complex type and winding trick.  Barnett 2/1/17
*/
{
  BIGINT k1min = -(ms / 2), k2min = -(mt / 2), k3min = -(mu / 2); // integer divide
  for (BIGINT j = 0; j < nj; ++j) {
    CPX a1   = (iflag > 0) ? exp(IMA * x[j]) : exp(-IMA * x[j]);
    CPX a2   = (iflag > 0) ? exp(IMA * y[j]) : exp(-IMA * y[j]);
    CPX a3   = (iflag > 0) ? exp(IMA * z[j]) : exp(-IMA * z[j]);
    CPX sp1  = pow(a1, (FLT)k1min);
    CPX sp2  = pow(a2, (FLT)k2min);
    CPX p3   = pow(a3, (FLT)k3min);
    CPX cc   = CPX(0, 0);
    BIGINT m = 0; // input pointer
    for (BIGINT m3 = 0; m3 < mu; ++m3) {
      CPX p2 = sp2;
      for (BIGINT m2 = 0; m2 < mt; ++m2) {
        CPX p1 = sp1;
        for (BIGINT m1 = 0; m1 < ms; ++m1) {
          cc += f[m++] * p1 * p2 * p3;
          p1 *= a1;
        }
        p2 *= a2;
      }
      p3 *= a3;
    }
    c[j] = cc;
  }
}

void dirft3d3(BIGINT nj, FLT *x, FLT *y, FLT *z, CPX *c, int iflag, BIGINT nk, FLT *s,
              FLT *t, FLT *u, CPX *f)
/* Direct computation of 3D type-3 nonuniform FFT. Interface same as finufft3d3
c               nj-1
c     f[k]  =   SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j] + u[k] z[j]))
c               j=0
c                    for k = 0, ..., nk-1
c  If iflag>0 the + sign is used, otherwise the - sign is used, in the
c  exponential. Uses C++ complex type. Simple brute force.  Barnett 2/1/17
*/
{
  for (BIGINT k = 0; k < nk; ++k) {
    CPX ss = (iflag > 0) ? IMA * s[k] : -IMA * s[k];
    CPX tt = (iflag > 0) ? IMA * t[k] : -IMA * t[k];
    CPX uu = (iflag > 0) ? IMA * u[k] : -IMA * u[k];
    f[k]   = CPX(0, 0);
    for (BIGINT j = 0; j < nj; ++j) f[k] += c[j] * exp(ss * x[j] + tt * y[j] + uu * z[j]);
  }
}
