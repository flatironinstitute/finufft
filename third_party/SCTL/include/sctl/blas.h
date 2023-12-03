/* Kernel Independent Fast Multipole Method
   Copyright (C) 2004 Lexing Ying, New York University

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, write to the Free
Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.  */

#ifndef _SCTL_BLAS_H_
#define _SCTL_BLAS_H_

extern "C" {
/*!  DGEMM  performs one of the matrix-matrix operations
*
*     C := alpha*op( A )*op( B ) + beta*C,
*
*  where  op( X ) is one of
*
*     op( X ) = X   or   op( X ) = X',
*
*  alpha and beta are scalars, and A, B and C are matrices, with op( A )
*  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
*  See http://www.netlib.org/blas/dgemm.f for more information.
*/
void sgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const float* ALPHA, const float* A, const int* LDA, const float* B, const int* LDB, const float* BETA, float* C, const int* LDC);
void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const double* ALPHA, const double* A, const int* LDA, const double* B, const int* LDB, const double* BETA, double* C, const int* LDC);
}

#endif
