/*
 * Minimal mex.h shim for compiling finufft.cpp (the mwrap-generated mex source)
 * outside of MATLAB, for use as a numbl backend (native shared library or
 * standalone WebAssembly).
 *
 * Implements just enough of the MATLAB mex C API to satisfy finufft.cpp:
 *   - mxArray as a tagged value (numeric / char / struct)
 *   - the interleaved-complex API (mxGetDoubles / mxGetComplexDoubles / ...)
 *   - mxCreateDoubleMatrix / mxCreateNumericMatrix / mxCreateString
 *   - struct introspection used by copy_finufft_opts
 *   - mexErrMsgTxt via setjmp/longjmp into the dispatch entry point
 *
 * MX_HAS_INTERLEAVED_COMPLEX is forced on so that finufft.cpp uses the
 * R2018a+ accessors. R2008OO is left undefined: class instances (the
 * MATLAB finufft_plan classdef) are unwrapped on the JS side and passed
 * as ordinary char-array pointers.
 */

#ifndef NUMBL_MEX_SHIM_MEX_H
#define NUMBL_MEX_SHIM_MEX_H

#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>

#ifndef MX_HAS_INTERLEAVED_COMPLEX
#define MX_HAS_INTERLEAVED_COMPLEX 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── basic typedefs ─────────────────────────────────────────────────────── */

typedef size_t mwSize;
typedef size_t mwIndex;
typedef ptrdiff_t mwSignedIndex;

typedef char mxChar;

typedef enum {
    mxUNKNOWN_CLASS = 0,
    mxCELL_CLASS,
    mxSTRUCT_CLASS,
    mxLOGICAL_CLASS,
    mxCHAR_CLASS,
    mxVOID_CLASS,
    mxDOUBLE_CLASS,
    mxSINGLE_CLASS,
    mxINT8_CLASS,
    mxUINT8_CLASS,
    mxINT16_CLASS,
    mxUINT16_CLASS,
    mxINT32_CLASS,
    mxUINT32_CLASS,
    mxINT64_CLASS,
    mxUINT64_CLASS,
    mxFUNCTION_CLASS
} mxClassID;

typedef enum {
    mxREAL = 0,
    mxCOMPLEX = 1
} mxComplexity;

typedef struct {
    double real;
    double imag;
} mxComplexDouble;

typedef struct {
    float real;
    float imag;
} mxComplexSingle;

/* mxArray is opaque to compiled callers; the layout lives in mex_shim.cpp. */
typedef struct mxArray_tag mxArray;

/* ── memory ─────────────────────────────────────────────────────────────── */

void* mxMalloc(size_t n);
void* mxCalloc(size_t n, size_t size);
void  mxFree(void* p);
void* mxRealloc(void* p, size_t n);

/* ── constructors ───────────────────────────────────────────────────────── */

mxArray* mxCreateDoubleMatrix(mwSize m, mwSize n, mxComplexity ComplexFlag);
mxArray* mxCreateNumericMatrix(mwSize m, mwSize n, mxClassID classid,
                                mxComplexity ComplexFlag);
mxArray* mxCreateString(const char* str);
mxArray* mxCreateDoubleScalar(double v);

/* ── property accessors ─────────────────────────────────────────────────── */

mwSize    mxGetM(const mxArray* a);
mwSize    mxGetN(const mxArray* a);
size_t    mxGetNumberOfElements(const mxArray* a);
mxClassID mxGetClassID(const mxArray* a);
int       mxIsComplex(const mxArray* a);
int       mxIsChar(const mxArray* a);
int       mxIsStruct(const mxArray* a);
int       mxIsDouble(const mxArray* a);

/* ── interleaved-complex data accessors (R2018a+) ───────────────────────── */

double*           mxGetDoubles(const mxArray* a);
mxComplexDouble*  mxGetComplexDoubles(const mxArray* a);
float*            mxGetSingles(const mxArray* a);
mxComplexSingle*  mxGetComplexSingles(const mxArray* a);

/* ── legacy accessors (kept compilable; treated as the real part) ───────── */

double*  mxGetPr(const mxArray* a);
double*  mxGetPi(const mxArray* a);
void*    mxGetData(const mxArray* a);
void*    mxGetImagData(const mxArray* a);

/* ── strings ────────────────────────────────────────────────────────────── */

mxChar*  mxGetChars(const mxArray* a);
int      mxGetString(const mxArray* a, char* buf, mwSize buflen);

/* ── struct introspection ───────────────────────────────────────────────── */

int          mxGetNumberOfFields(const mxArray* a);
const char*  mxGetFieldNameByNumber(const mxArray* a, int idx);
mxArray*     mxGetFieldByNumber(const mxArray* a, mwIndex elem, int idx);
mxArray*     mxGetField(const mxArray* a, mwIndex elem, const char* fieldname);

/* mxGetProperty is unused without R2008OO but provided for completeness. */
mxArray*     mxGetProperty(const mxArray* a, mwIndex elem, const char* name);

/* ── error / printing / locking ─────────────────────────────────────────── */

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))
#endif
void mexErrMsgTxt(const char* msg);

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn,format(printf,2,3)))
#endif
void mexErrMsgIdAndTxt(const char* id, const char* fmt, ...);

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf,1,2)))
#endif
int  mexPrintf(const char* fmt, ...);

void mexLock(void);
void mexUnlock(void);
int  mexEvalString(const char* str);

#ifdef __cplusplus
}
#endif

#endif /* NUMBL_MEX_SHIM_MEX_H */
