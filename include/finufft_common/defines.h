#pragma once

/* IMPORTANT: for Windows compilers, you should add a line
        #define FINUFFT_DLL
   here if you are compiling/using FINUFFT as a DLL,
   in order to do the proper importing/exporting, or
   alternatively compile with -DFINUFFT_DLL or the equivalent
   command-line flag.  This is not necessary under MinGW/Cygwin, where
   libtool does the imports/exports automatically.
   Alternatively use include(GenerateExportHeader) and
   generate_export_header(finufft) to auto generate an header containing
   these defines.The main reason is that if msvc changes the way it deals
   with it in the future we just need to update cmake for it to work
   instead of having a check on the msvc version. */
#if defined(FINUFFT_DLL)
#if defined(_WIN32) || defined(__WIN32__)
#if defined(dll_EXPORTS)
#define FINUFFT_EXPORT __declspec(dllexport)
#else
#define FINUFFT_EXPORT __declspec(dllimport)
#endif
#else
#define FINUFFT_EXPORT __attribute__((visibility("default")))
#endif
#else
#define FINUFFT_EXPORT
#endif

/* specify calling convention (Windows only)
   The cdecl calling convention is actually not the default in all but a very
   few C/C++ compilers.
   If the user code changes the default compiler calling convention, may need
   this when generating DLL. */
#if defined(_WIN32) || defined(__WIN32__)
#define FINUFFT_CDECL __cdecl
#else
#define FINUFFT_CDECL
#endif

// common function attributes
#if defined(_MSC_VER)
#define FINUFFT_ALWAYS_INLINE __forceinline
#define FINUFFT_NEVER_INLINE  __declspec(noinline)
#define FINUFFT_RESTRICT      __restrict
#define FINUFFT_UNREACHABLE   __assume(0)
#define FINUFFT_UNLIKELY(x)   (x)
#define FINUFFT_LIKELY(x)     (x)
#elif defined(__GNUC__) || defined(__clang__)
#define FINUFFT_ALWAYS_INLINE __attribute__((always_inline)) inline
#define FINUFFT_NEVER_INLINE  __attribute__((noinline))
#define FINUFFT_RESTRICT      __restrict__
#define FINUFFT_UNREACHABLE   __builtin_unreachable()
#define FINUFFT_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#define FINUFFT_LIKELY(x)     __builtin_expect(!!(x), 1)
#else
#define FINUFFT_ALWAYS_INLINE inline
#define FINUFFT_NEVER_INLINE
#define FINUFFT_RESTRICT
#define FINUFFT_UNREACHABLE
#define FINUFFT_UNLIKELY(x) (x)
#define FINUFFT_LIKELY(x)   (x)
#endif
