if(FINUFFT_USE_CPU)
    if(WIN32)
        install(TARGETS finufft LIBRARY DESTINATION finufft RUNTIME DESTINATION finufft)
    else()
        install(TARGETS finufft LIBRARY DESTINATION finufft)
    endif()
endif()

if(FINUFFT_USE_CUDA)
    install(TARGETS cufinufft LIBRARY DESTINATION cufinufft)
endif()

# Warn if the user invokes CMake directly
if(NOT SKBUILD)
    message(
        WARNING
        "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install python/finufft
   or
   $ pip install python/cufinufft
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install finufft scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files."
    )
endif()
