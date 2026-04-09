/*
 * Implementation of the minimal mex shim declared in mex_shim/mex.h, plus
 * the marshalling entry points exposed to numbl's JS user-function loader.
 *
 * The shim backs mxArray with simple heap-allocated tagged structs that
 * own their data.  finufft.cpp's mexFunction is invoked through
 * mex_dispatch(); errors raised via mexErrMsgTxt longjmp out to the
 * dispatch entry, which then surfaces the message to JS.
 */

#include "mex_shim/mex.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <csetjmp>
#include <cmath>

/* ── mxArray representation ────────────────────────────────────────────── */

struct StructField {
    char* name;
    mxArray* value;
};

struct mxArray_tag {
    mxClassID classID;
    int       isComplex;
    mwSize    m;
    mwSize    n;

    /* numeric data: m*n elements (real) or m*n mxComplexDouble (complex) */
    void* data;

    /* char arrays: m*n bytes (no NUL terminator stored) */
    /* (uses `data` field) */

    /* struct: nfields entries; m,n must be 1 */
    int           nfields;
    StructField*  fields;
};

static mxArray* alloc_mxArray() {
    mxArray* a = (mxArray*)std::calloc(1, sizeof(mxArray));
    a->classID = mxUNKNOWN_CLASS;
    return a;
}

static void mxArray_free_recursive(mxArray* a) {
    if (!a) return;
    if (a->classID == mxSTRUCT_CLASS) {
        for (int i = 0; i < a->nfields; ++i) {
            std::free(a->fields[i].name);
            mxArray_free_recursive(a->fields[i].value);
        }
        std::free(a->fields);
    } else if (a->data) {
        std::free(a->data);
    }
    std::free(a);
}

/* ── error handling via longjmp out of mexFunction ─────────────────────── */

static jmp_buf g_jmp;
static int     g_jmp_active = 0;
static char    g_err_msg[1024];

static void set_err(const char* msg) {
    if (msg) {
        std::strncpy(g_err_msg, msg, sizeof(g_err_msg) - 1);
        g_err_msg[sizeof(g_err_msg) - 1] = '\0';
    } else {
        g_err_msg[0] = '\0';
    }
}

/* ── memory ─────────────────────────────────────────────────────────────── */

extern "C" {

void* mxMalloc(size_t n)              { return std::malloc(n); }
void* mxCalloc(size_t n, size_t size) { return std::calloc(n, size); }
void  mxFree(void* p)                 { std::free(p); }
void* mxRealloc(void* p, size_t n)    { return std::realloc(p, n); }

/* ── constructors ───────────────────────────────────────────────────────── */

mxArray* mxCreateNumericMatrix(mwSize m, mwSize n, mxClassID classid,
                                mxComplexity ComplexFlag) {
    mxArray* a = alloc_mxArray();
    a->classID = classid;
    a->isComplex = (ComplexFlag == mxCOMPLEX) ? 1 : 0;
    a->m = m;
    a->n = n;
    size_t elt_size;
    switch (classid) {
        case mxDOUBLE_CLASS: elt_size = sizeof(double); break;
        case mxSINGLE_CLASS: elt_size = sizeof(float);  break;
        case mxINT64_CLASS:
        case mxUINT64_CLASS: elt_size = sizeof(int64_t); break;
        case mxINT32_CLASS:
        case mxUINT32_CLASS: elt_size = sizeof(int32_t); break;
        default:             elt_size = sizeof(double); break;
    }
    if (a->isComplex) elt_size *= 2;
    size_t bytes = m * n * elt_size;
    a->data = bytes ? std::calloc(1, bytes) : nullptr;
    return a;
}

mxArray* mxCreateDoubleMatrix(mwSize m, mwSize n, mxComplexity ComplexFlag) {
    return mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS, ComplexFlag);
}

mxArray* mxCreateDoubleScalar(double v) {
    mxArray* a = mxCreateDoubleMatrix(1, 1, mxREAL);
    *(double*)a->data = v;
    return a;
}

mxArray* mxCreateString(const char* str) {
    mxArray* a = alloc_mxArray();
    a->classID = mxCHAR_CLASS;
    a->isComplex = 0;
    size_t len = str ? std::strlen(str) : 0;
    a->m = len ? 1 : 0;
    a->n = len;
    a->data = std::calloc(len + 1, 1);
    if (len) std::memcpy(a->data, str, len);
    return a;
}

/* ── property accessors ─────────────────────────────────────────────────── */

mwSize    mxGetM(const mxArray* a)      { return a ? a->m : 0; }
mwSize    mxGetN(const mxArray* a)      { return a ? a->n : 0; }
size_t    mxGetNumberOfElements(const mxArray* a) {
    return a ? a->m * a->n : 0;
}
mxClassID mxGetClassID(const mxArray* a) {
    return a ? a->classID : mxUNKNOWN_CLASS;
}
int       mxIsComplex(const mxArray* a) { return a && a->isComplex; }
int       mxIsChar(const mxArray* a)    { return a && a->classID == mxCHAR_CLASS; }
int       mxIsStruct(const mxArray* a)  { return a && a->classID == mxSTRUCT_CLASS; }
int       mxIsDouble(const mxArray* a)  { return a && a->classID == mxDOUBLE_CLASS; }

/* ── data accessors ─────────────────────────────────────────────────────── */

double*           mxGetDoubles(const mxArray* a)        { return a ? (double*)a->data : nullptr; }
mxComplexDouble*  mxGetComplexDoubles(const mxArray* a) { return a ? (mxComplexDouble*)a->data : nullptr; }
float*            mxGetSingles(const mxArray* a)        { return a ? (float*)a->data : nullptr; }
mxComplexSingle*  mxGetComplexSingles(const mxArray* a) { return a ? (mxComplexSingle*)a->data : nullptr; }
double*           mxGetPr(const mxArray* a)             { return a ? (double*)a->data : nullptr; }
double*           mxGetPi(const mxArray* /*a*/)         { return nullptr; }
void*             mxGetData(const mxArray* a)           { return a ? a->data : nullptr; }
void*             mxGetImagData(const mxArray* /*a*/)   { return nullptr; }
mxChar*           mxGetChars(const mxArray* a)          { return a ? (mxChar*)a->data : nullptr; }

int mxGetString(const mxArray* a, char* buf, mwSize buflen) {
    if (!a || !buf || buflen == 0) return 1;
    size_t n = a->m * a->n;
    if (n + 1 > buflen) n = buflen - 1;
    if (n && a->data) std::memcpy(buf, a->data, n);
    buf[n] = '\0';
    return 0;
}

/* ── struct introspection ──────────────────────────────────────────────── */

int mxGetNumberOfFields(const mxArray* a) {
    if (!a || a->classID != mxSTRUCT_CLASS) return 0;
    return a->nfields;
}

const char* mxGetFieldNameByNumber(const mxArray* a, int idx) {
    if (!a || a->classID != mxSTRUCT_CLASS) return nullptr;
    if (idx < 0 || idx >= a->nfields) return nullptr;
    return a->fields[idx].name;
}

mxArray* mxGetFieldByNumber(const mxArray* a, mwIndex /*elem*/, int idx) {
    if (!a || a->classID != mxSTRUCT_CLASS) return nullptr;
    if (idx < 0 || idx >= a->nfields) return nullptr;
    return a->fields[idx].value;
}

mxArray* mxGetField(const mxArray* a, mwIndex elem, const char* fieldname) {
    if (!a || a->classID != mxSTRUCT_CLASS || !fieldname) return nullptr;
    for (int i = 0; i < a->nfields; ++i) {
        if (a->fields[i].name && std::strcmp(a->fields[i].name, fieldname) == 0)
            return mxGetFieldByNumber(a, elem, i);
    }
    return nullptr;
}

mxArray* mxGetProperty(const mxArray* /*a*/, mwIndex /*elem*/, const char* /*name*/) {
    /* R2008OO not enabled in finufft.cpp; class instances are unwrapped JS-side. */
    return nullptr;
}

/* ── error / printing / locking ─────────────────────────────────────────── */

void mexErrMsgTxt(const char* msg) {
    set_err(msg && *msg ? msg : "mexErrMsgTxt");
    if (g_jmp_active) longjmp(g_jmp, 1);
    /* If called outside a dispatch we have nowhere to bail to — abort. */
    std::abort();
}

void mexErrMsgIdAndTxt(const char* /*id*/, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt ? fmt : "", ap);
    va_end(ap);
    mexErrMsgTxt(buf);
}

int mexPrintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = std::vfprintf(stdout, fmt ? fmt : "", ap);
    va_end(ap);
    return r;
}

void mexLock(void) { /* no-op */ }
void mexUnlock(void) { /* no-op */ }

int mexEvalString(const char* /*str*/) {
    /* finufft_mex_setup() calls mexEvalString("fft(1:8);") to force MATLAB
     * to initialize FFTW.  Outside MATLAB this is a no-op. */
    return 0;
}

} /* extern "C" */

/* ── mex dispatch entry point + JS-side helpers ────────────────────────── */

/* finufft.cpp defines mexFunction without an extern "C" wrapper, so it
 * carries C++ linkage.  Match that here. */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

#if defined(__wasm__)
#define EXPORT(name) __attribute__((export_name(#name), used))
#else
#define EXPORT(name) __attribute__((visibility("default"), used))
#endif

extern "C" {

/* General-purpose heap helpers (also used by JS to copy data into wasm). */
EXPORT(my_malloc) void* my_malloc(int size) { return std::malloc(size); }
EXPORT(my_free)   void  my_free(void* ptr)  { std::free(ptr); }

/* Allocate / free an mxArray*[] (used for prhs and plhs). */
EXPORT(mex_alloc_args) mxArray** mex_alloc_args(int n) {
    if (n <= 0) n = 1;
    return (mxArray**)std::calloc((size_t)n, sizeof(mxArray*));
}

EXPORT(mex_free_args) void mex_free_args(mxArray** arr) {
    std::free(arr);
}

EXPORT(mex_set_arg) void mex_set_arg(mxArray** arr, int idx, mxArray* val) {
    arr[idx] = val;
}

EXPORT(mex_get_arg) mxArray* mex_get_arg(mxArray** arr, int idx) {
    return arr[idx];
}

/* Construction helpers. */
EXPORT(mex_make_double_scalar) mxArray* mex_make_double_scalar(double v) {
    return mxCreateDoubleScalar(v);
}

/* Allocate a real m×n double matrix and copy `data` (m*n elements) into it.
 * Pass data=NULL (or m*n=0) to leave the buffer zeroed/empty. */
EXPORT(mex_make_real_matrix)
mxArray* mex_make_real_matrix(int m, int n, const double* data) {
    mxArray* a = mxCreateDoubleMatrix((mwSize)m, (mwSize)n, mxREAL);
    if (data && m * n > 0) {
        std::memcpy(a->data, data, sizeof(double) * (size_t)m * (size_t)n);
    }
    return a;
}

/* Allocate a complex m×n double matrix from separate real/imag buffers.
 * Either pointer may be NULL to mean "all zeros". */
EXPORT(mex_make_complex_matrix)
mxArray* mex_make_complex_matrix(int m, int n,
                                  const double* re, const double* im) {
    mxArray* a = mxCreateDoubleMatrix((mwSize)m, (mwSize)n, mxCOMPLEX);
    mxComplexDouble* dst = (mxComplexDouble*)a->data;
    size_t total = (size_t)m * (size_t)n;
    for (size_t i = 0; i < total; ++i) {
        dst[i].real = re ? re[i] : 0.0;
        dst[i].imag = im ? im[i] : 0.0;
    }
    return a;
}

EXPORT(mex_make_string) mxArray* mex_make_string(const char* s) {
    return mxCreateString(s);
}

EXPORT(mex_make_struct) mxArray* mex_make_struct(int nfields) {
    mxArray* a = alloc_mxArray();
    a->classID = mxSTRUCT_CLASS;
    a->m = 1;
    a->n = 1;
    a->nfields = nfields;
    a->fields = (StructField*)std::calloc((size_t)(nfields > 0 ? nfields : 1),
                                          sizeof(StructField));
    return a;
}

EXPORT(mex_struct_set_field)
void mex_struct_set_field(mxArray* s, int idx, const char* name, mxArray* value) {
    if (!s || s->classID != mxSTRUCT_CLASS) return;
    if (idx < 0 || idx >= s->nfields) return;
    std::free(s->fields[idx].name);
    if (s->fields[idx].value) mxArray_free_recursive(s->fields[idx].value);
    if (name) {
        size_t len = std::strlen(name) + 1;
        s->fields[idx].name = (char*)std::malloc(len);
        std::memcpy(s->fields[idx].name, name, len);
    } else {
        s->fields[idx].name = nullptr;
    }
    s->fields[idx].value = value;
}

/* Output mxArray inspection. */
EXPORT(mex_get_classid)    int mex_get_classid(mxArray* a)    { return a ? (int)a->classID : 0; }
EXPORT(mex_get_m)          int mex_get_m(mxArray* a)          { return a ? (int)a->m : 0; }
EXPORT(mex_get_n)          int mex_get_n(mxArray* a)          { return a ? (int)a->n : 0; }
EXPORT(mex_get_is_complex) int mex_get_is_complex(mxArray* a) { return a ? a->isComplex : 0; }

/* Output reading helpers — copy data out into a caller-provided buffer.
 * For wasm callers, the buffer is in linear memory (allocated via my_malloc).
 * For native callers, koffi marshals JS typed arrays automatically. */
EXPORT(mex_read_double_scalar) double mex_read_double_scalar(mxArray* a) {
    if (!a || !a->data) return 0.0;
    return *(double*)a->data;
}

EXPORT(mex_read_real)
void mex_read_real(mxArray* a, double* out) {
    if (!a || !a->data || !out) return;
    std::memcpy(out, a->data, sizeof(double) * (size_t)a->m * (size_t)a->n);
}

EXPORT(mex_read_complex)
void mex_read_complex(mxArray* a, double* out_re, double* out_im) {
    if (!a || !a->data) return;
    mxComplexDouble* src = (mxComplexDouble*)a->data;
    size_t n = (size_t)a->m * (size_t)a->n;
    for (size_t i = 0; i < n; ++i) {
        if (out_re) out_re[i] = src[i].real;
        if (out_im) out_im[i] = src[i].imag;
    }
}

/* Read string-typed mxArray contents into a caller buffer.  Returns the
 * number of bytes written (excluding the trailing NUL). */
EXPORT(mex_read_string)
int mex_read_string(mxArray* a, char* buf, int buflen) {
    if (!a || a->classID != mxCHAR_CLASS || !buf || buflen <= 0) {
        if (buf && buflen > 0) buf[0] = '\0';
        return 0;
    }
    size_t n = (size_t)a->m * (size_t)a->n;
    if ((int)n > buflen - 1) n = (size_t)(buflen - 1);
    if (n && a->data) std::memcpy(buf, a->data, n);
    buf[n] = '\0';
    return (int)n;
}

EXPORT(mex_free_array) void mex_free_array(mxArray* a) {
    mxArray_free_recursive(a);
}

/* Dispatch into mexFunction with longjmp-based error capture.
 * Returns 0 on success, 1 on error.  Use mex_get_error() to read the message. */
EXPORT(mex_dispatch)
int mex_dispatch(int nlhs, mxArray** plhs, int nrhs, mxArray** prhs) {
    g_err_msg[0] = '\0';
    g_jmp_active = 1;
    int rc;
    if (setjmp(g_jmp) == 0) {
        mexFunction(nlhs, plhs, nrhs, (const mxArray**)prhs);
        rc = 0;
    } else {
        rc = 1;
    }
    g_jmp_active = 0;
    return rc;
}

EXPORT(mex_get_error) const char* mex_get_error(void) {
    return g_err_msg;
}

} /* extern "C" */
