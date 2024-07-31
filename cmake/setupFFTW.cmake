cpmaddpackage(
  NAME
  findfftw
  GIT_REPOSITORY
  "https://github.com/egpbos/findFFTW.git"
  GIT_TAG
  "master"
  EXCLUDE_FROM_ALL
  YES
  GIT_SHALLOW
  YES)

list(APPEND CMAKE_MODULE_PATH "${findfftw_SOURCE_DIR}")

if(FINUFFT_FFTW_LIBRARIES STREQUAL DEFAULT OR FINUFFT_FFTW_LIBRARIES STREQUAL
                                              DOWNLOAD)
  find_package(FFTW)
  if((NOT FFTW_FOUND) OR (FINUFFT_FFTW_LIBRARIES STREQUAL DOWNLOAD))
    if(FINUFFT_FFTW_SUFFIX STREQUAL THREADS)
      set(FINUFFT_USE_THREADS ON)
    else()
      set(FINUFFT_USE_THREADS OFF)
    endif()
    cpmaddpackage(
      NAME
      fftw3
      OPTIONS
      "ENABLE_SSE2 ON"
      "ENABLE_AVX ON"
      "ENABLE_AVX2 ON"
      "BUILD_TESTS OFF"
      "BUILD_SHARED_LIBS OFF"
      "ENABLE_THREADS ${FINUFFT_USE_THREADS}"
      "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
      URL
      "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
      URL_HASH
      "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
      EXCLUDE_FROM_ALL
      YES
      GIT_SHALLOW
      YES)

    cpmaddpackage(
      NAME
      fftw3f
      OPTIONS
      "ENABLE_SSE2 ON"
      "ENABLE_AVX ON"
      "ENABLE_AVX2 ON"
      "ENABLE_FLOAT ON"
      "BUILD_TESTS OFF"
      "BUILD_SHARED_LIBS OFF"
      "ENABLE_THREADS ${FINUFFT_USE_THREADS}"
      "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
      URL
      "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
      URL_HASH
      "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
      EXCLUDE_FROM_ALL
      YES
      GIT_SHALLOW
      YES)
    set(FINUFFT_FFTW_LIBRARIES fftw3 fftw3f)
    if(FINUFFT_USE_THREADS)
      list(APPEND FINUFFT_FFTW_LIBRARIES fftw3_threads fftw3f_threads)
    elseif(FINUFFT_USE_OPENMP)
      list(APPEND FINUFFT_FFTW_LIBRARIES fftw3_omp fftw3f_omp)
    endif()

    foreach(element IN LISTS FINUFFT_FFTW_LIBRARIES)
      set_target_properties(
        ${element}
        PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
                   POSITION_INDEPENDENT_CODE ${FINUFFT_SHARED_LINKING})
    endforeach()

    target_include_directories(
      fftw3 PUBLIC $<BUILD_INTERFACE:${fftw3_SOURCE_DIR}/api>)

  else()
    set(FINUFFT_FFTW_LIBRARIES
        "FFTW::Float" "FFTW::Double" "FFTW::Float${FINUFFT_FFTW_SUFFIX}"
        "FFTW::Double${FINUFFT_FFTW_SUFFIX}")
  endif()
endif()
