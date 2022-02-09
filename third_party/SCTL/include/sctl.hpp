// Scientific Computing Template Library

#ifndef _SCTL_HPP_
#define _SCTL_HPP_

#include <sctl/common.hpp>

// Import PVFMM preprocessor macro definitions
#ifdef SCTL_HAVE_PVFMM
#ifndef SCTL_HAVE_MPI
#define SCTL_HAVE_MPI
#endif
#include "pvfmm_config.h"
#if defined(PVFMM_QUAD_T) && !defined(SCTL_QUAD_T)
#define SCTL_QUAD_T PVFMM_QUAD_T
#endif
#endif

// Math utilities
#include SCTL_INCLUDE(math_utils.hpp)

// FMM wrapper
#include SCTL_INCLUDE(fmm-wrapper.hpp)

// Boundary Integrals
#include SCTL_INCLUDE(boundary_integral.hpp)
#include SCTL_INCLUDE(slender_element.hpp)
#include SCTL_INCLUDE(quadrule.hpp)

// ODE solver
#include SCTL_INCLUDE(ode-solver.hpp)

// Tensor
#include SCTL_INCLUDE(tensor.hpp)

// Tree
#include SCTL_INCLUDE(tree.hpp)
#include SCTL_INCLUDE(vtudata.hpp)

// MPI Wrapper
#include SCTL_INCLUDE(comm.hpp)

// Memory Manager, Iterators
#include SCTL_INCLUDE(mem_mgr.hpp)

// Vector
#include SCTL_INCLUDE(vector.hpp)

// Matrix, Permutation operators
#include SCTL_INCLUDE(matrix.hpp)

// Template vector intrinsics (new)
#include SCTL_INCLUDE(vec.hpp)
#include SCTL_INCLUDE(vec-test.hpp)

// OpenMP merge-sort and scan
#include SCTL_INCLUDE(ompUtils.hpp)

// Parallel solver
#include SCTL_INCLUDE(parallel_solver.hpp)

// Chebyshev basis
#include SCTL_INCLUDE(cheb_utils.hpp)

// Morton
#include SCTL_INCLUDE(morton.hpp)

// Spherical Harmonics
#include SCTL_INCLUDE(sph_harm.hpp)

#include SCTL_INCLUDE(fft_wrapper.hpp)

#include SCTL_INCLUDE(legendre_rule.hpp)

// Profiler
#include SCTL_INCLUDE(profile.hpp)

// Print stack trace
#include SCTL_INCLUDE(stacktrace.h)
const int sgh = SCTL_NAMESPACE::SetSigHandler(); // Set signal handler

// Boundary quadrature, Kernel functions
#include SCTL_INCLUDE(kernel_functions.hpp)
#include SCTL_INCLUDE(boundary_quadrature.hpp)

#endif //_SCTL_HPP_
