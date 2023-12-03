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

#ifndef _SCTL_LAPACK_H_
#define _SCTL_LAPACK_H_

// EXTERN_C_BEGIN
extern "C" {
void sgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N, float* A, const int* LDA, float* S, float* U, const int* LDU, float* VT, const int* LDVT, float* WORK, const int* LWORK, int* INFO);
/*!    DGESVD computes the singular value decomposition (SVD) of a real
 *  M-by-N matrix A, optionally computing the left and/or right singular
 *  vectors. The SVD is written
 *
 *       A = U * SIGMA * transpose(V)
 *
 *  where SIGMA is an M-by-N matrix which is zero except for its
 *  min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 *  V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
 *  are the singular values of A; they are real and non-negative, and
 *  are returned in descending order.  The first min(m,n) columns of
 *  U and V are the left and right singular vectors of A.
 *
 * See http://www.netlib.org/lapack/double/dgesvd.f for more information
 */
void dgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N, double* A, const int* LDA, double* S, double* U, const int* LDU, double* VT, const int* LDVT, double* WORK, const int* LWORK, int* INFO);
}
// EXTERN_C_END

#endif
