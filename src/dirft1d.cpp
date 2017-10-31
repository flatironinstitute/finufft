#include "dirft.h"
#include <iostream>

// This is basically a port of dirft1d.f from CMCL package, except with
// the 1/nj prefactors for type-1 removed.

void dirft1d1(INT nj,FLT* x,CPX* c,int iflag,INT ms, CPX* f)
/* Direct computation of 1D type-1 nonuniform FFT. Interface same as finufft1d1.
c                  nj-1
c     f[k1]    =   SUM  c[j] exp(+-i k1 x[j]) 
c                  j=0
c
c     for -ms/2 <= k1 <= (ms-1)/2.
c     The output array is in increasing k1 ordering. If iflag>0 the + sign is
c     used, otherwise the - sign is used, in the exponential.
*  Uses C++ complex type and winding trick.  Barnett 1/25/17
*/
{
  INT kmin = -(ms/2);                   // integer divide
  for (INT m=0;m<ms;++m) f[m] = (0,0);  // it knows f is complex type
  for (INT j=0;j<nj;++j) {
    CPX a = (iflag>0) ? exp(ima*x[j]) : exp(-ima*x[j]);
    CPX p = pow(a,(FLT)kmin);   // starting phase for most neg freq
    CPX cc = c[j];              // no 1/nj prefac
    for (INT m=0;m<ms;++m) {
      f[m] += cc * p;
      p *= a;
    }
  }
}

void dirft1d2(INT nj,FLT* x,CPX* c,int iflag,INT ms, CPX* f)
/* Direct computation of 1D type-2 nonuniform FFT. Interface same as finufft1d2
c
c     c[j] = SUM   f[k1] exp(+-i k1 x[j]) 
c             k1  
c                            for j = 0,...,nj-1
c     where sum is over -ms/2 <= k1 <= (ms-1)/2.
c     The input array is in increasing k1 ordering. If iflag>0 the + sign is
c     used, otherwise the - sign is used, in the exponential.
*  Uses C++ complex type and winding trick.  Barnett 1/25/17
*/
{
  INT kmin = -(ms/2);                     // integer divide
  for (INT j=0;j<nj;++j) {
    CPX a = (iflag>0) ? exp(ima*x[j]) : exp(-ima*x[j]);
    CPX p = pow(a,(FLT)kmin);   // starting phase for most neg freq
    CPX cc = (0,0);
    for (INT m=0;m<ms;++m) {
      cc += f[m] * p;
      p *= a;
    }
    c[j] = cc;
  }
}

void dirft1d3(INT nj,FLT* x,CPX* c,int iflag,INT nk, FLT* s, CPX* f)
/* Direct computation of 1D type-3 nonuniform FFT. Interface same as finufft1d3
c              nj-1
c     f[k]  =  SUM   c[j] exp(+-i s[k] x[j]) 
c              j=0                   
c                    for k = 0, ..., nk-1
c  If iflag>0 the + sign is used, otherwise the - sign is used, in the
c  exponential. Uses C++ complex type. Simple brute force.  Barnett 1/25/17
*/
{
  for (INT k=0;k<nk;++k) {
    CPX ss = (iflag>0) ? ima*s[k] : -ima*s[k];
    f[k] = (0,0);
    for (INT j=0;j<nj;++j)
      f[k] += c[j] * exp(ss*x[j]);
  }
}
