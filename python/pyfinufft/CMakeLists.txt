if (NOT SKBUILD)
    message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)

set(NANOBIND_DOWNLOAD_VERSION v1.8.0)

CPMAddPackage(
        NAME nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind
        GIT_TAG ${NANOBIND_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
)

nanobind_add_module(pyfinufft_ext STABLE_ABI NB_STATIC pyfinufft_ext.cpp)
target_link_libraries(pyfinufft_ext PRIVATE finufft_static)
install(TARGETS pyfinufft_ext LIBRARY DESTINATION .)
