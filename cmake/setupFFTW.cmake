CPMAddPackage(
    NAME
    findfftw
    GIT_REPOSITORY
    "https://github.com/egpbos/findFFTW.git"
    GIT_TAG
    "master"
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
    YES
    GIT_SHALLOW
    YES
)

list(APPEND CMAKE_MODULE_PATH "${findfftw_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

if(FINUFFT_FFTW_LIBRARIES STREQUAL DEFAULT OR FINUFFT_FFTW_LIBRARIES STREQUAL DOWNLOAD)
    set(FINUFFT_FFTW_DL_VERSION "${FFTW_VERSION}")
    find_package(FFTW)
    if((NOT FFTW_FOUND) OR (FINUFFT_FFTW_LIBRARIES STREQUAL DOWNLOAD))
        # find_package(FFTW) sets FFTW_VERSION to the installed version; restore the pin.
        set(FFTW_VERSION "${FINUFFT_FFTW_DL_VERSION}")
        if(FINUFFT_FFTW_SUFFIX STREQUAL THREADS)
            set(FINUFFT_USE_THREADS ON)
        else()
            set(FINUFFT_USE_THREADS OFF)
        endif()
        # 3.3.11 is out, but its CMake build breaks this download; stay on 3.3.10.
        CPMAddPackage(
            NAME
            fftw3
            URL
            "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
            URL_HASH
            "SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"
            EXCLUDE_FROM_ALL
            YES
            SYSTEM
            YES
            OPTIONS
            "ENABLE_SSE2 ON"
            "ENABLE_AVX ON"
            "ENABLE_AVX2 ON"
            "BUILD_TESTS OFF"
            "BUILD_SHARED_LIBS OFF"
            "ENABLE_THREADS ${FINUFFT_USE_THREADS}"
            "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
            "CMAKE_POLICY_VERSION_MINIMUM 3.10"
        )

        CPMAddPackage(
            NAME
            fftw3f
            URL
            "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
            URL_HASH
            "SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"
            EXCLUDE_FROM_ALL
            YES
            SYSTEM
            YES
            OPTIONS
            "ENABLE_SSE2 ON"
            "ENABLE_AVX ON"
            "ENABLE_AVX2 ON"
            "ENABLE_FLOAT ON"
            "BUILD_TESTS OFF"
            "BUILD_SHARED_LIBS OFF"
            "ENABLE_THREADS ${FINUFFT_USE_THREADS}"
            "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
            "CMAKE_POLICY_VERSION_MINIMUM 3.10"
        )
        set(FINUFFT_FFTW_LIBRARIES fftw3 fftw3f)
        if(FINUFFT_USE_THREADS)
            list(APPEND FINUFFT_FFTW_LIBRARIES fftw3_threads fftw3f_threads)
        elseif(FINUFFT_USE_OPENMP)
            list(APPEND FINUFFT_FFTW_LIBRARIES fftw3_omp fftw3f_omp)
        endif()

        if(NOT TARGET fftw3)
            message(
                FATAL_ERROR
                "FINUFFT could not fetch FFTW ${FFTW_VERSION} from http://www.fftw.org "
                "(FETCHCONTENT_FULLY_DISCONNECTED=${FETCHCONTENT_FULLY_DISCONNECTED}). "
                "Install FFTW where find_package(FFTW) can see it, or allow the download."
            )
        endif()

        foreach(element IN LISTS FINUFFT_FFTW_LIBRARIES)
            set_target_properties(
                ${element}
                PROPERTIES
                    POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
                    MSVC_DEBUG_INFORMATION_FORMAT Embedded
            )
        endforeach()

        target_include_directories(fftw3 PUBLIC $<BUILD_INTERFACE:${fftw3_SOURCE_DIR}/api>)
        set(FINUFFT_FFT_BUNDLE ${FINUFFT_FFTW_LIBRARIES})
    else()
        # link against single thread fftw
        set(FINUFFT_FFTW_LIBRARIES "FFTW::Float" "FFTW::Double")
        # default behavior
        if(FINUFFT_FFTW_SUFFIX STREQUAL "DEFAULT")
            if(FINUFFT_USE_OPENMP)
                list(APPEND FINUFFT_FFTW_LIBRARIES "FFTW::FloatOpenMP" "FFTW::DoubleOpenMP")
            endif()
        else()
            # user override
            list(APPEND FINUFFT_FFTW_LIBRARIES "FFTW::Float${FINUFFT_FFTW_SUFFIX}" "FFTW::Double${FINUFFT_FFTW_SUFFIX}")
        endif()
        set(FINUFFT_FFTW_FIND_MODULE "${findfftw_SOURCE_DIR}/FindFFTW.cmake")
    endif()
endif()

add_library(finufft_fftlibs INTERFACE)
target_link_libraries(finufft_fftlibs INTERFACE ${FINUFFT_FFTW_LIBRARIES})

if(FINUFFT_ENABLE_INSTALL AND FINUFFT_STATIC_LINKING AND NOT FINUFFT_FFT_BUNDLE AND NOT FINUFFT_FFTW_FIND_MODULE)
    message(
        WARNING
        "FINUFFT_FFTW_LIBRARIES=${FINUFFT_FFTW_LIBRARIES} is supplied by hand, so the installed "
        "static package cannot export it: a consumer of finufft::finufft has to link the same FFTW itself."
    )
endif()

set(FINUFFT_FFT_BUNDLE "${FINUFFT_FFT_BUNDLE}" PARENT_SCOPE)
set(FINUFFT_FFTW_FIND_MODULE "${FINUFFT_FFTW_FIND_MODULE}" PARENT_SCOPE)
