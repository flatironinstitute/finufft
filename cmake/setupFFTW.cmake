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
    find_package(FFTW)
    if((NOT FFTW_FOUND) OR (FINUFFT_FFTW_LIBRARIES STREQUAL DOWNLOAD))
        if(FINUFFT_FFTW_SUFFIX STREQUAL THREADS)
            set(FINUFFT_USE_THREADS ON)
        else()
            set(FINUFFT_USE_THREADS OFF)
        endif()
        CPMAddPackage(
            NAME
            fftw3
            URL
            "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
            URL_HASH
            "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
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
            "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
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

        foreach(element IN LISTS FINUFFT_FFTW_LIBRARIES)
            set_target_properties(
                ${element}
                PROPERTIES
                    POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
                    MSVC_DEBUG_INFORMATION_FORMAT Embedded
            )
        endforeach()

        target_include_directories(fftw3 PUBLIC $<BUILD_INTERFACE:${fftw3_SOURCE_DIR}/api>)
        # FINUFFT builds these archives itself, so a static install ships them
        # inside finufftTargets (see the install block in the top-level CMakeLists).
        set(FINUFFT_FFT_EXPORT_TARGETS ${FINUFFT_FFTW_LIBRARIES})
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
        # FFTW::* are imported targets of the system FFTW, which no export set can
        # carry. A static install ships this FindFFTW module next to the package
        # config instead, and finufftConfig.cmake re-runs it to recreate them.
        set(FINUFFT_FFT_FIND_MODULE "${findfftw_SOURCE_DIR}/FindFFTW.cmake")
    endif()
endif()

add_library(finufft_fftlibs INTERFACE)
target_link_libraries(finufft_fftlibs INTERFACE ${FINUFFT_FFTW_LIBRARIES})

# A user-supplied FINUFFT_FFTW_LIBRARIES leaves both variables empty: that FFTW is
# the user's to reproduce, so the install interface stays silent about it.
set(FINUFFT_FFT_EXPORT_TARGETS "${FINUFFT_FFT_EXPORT_TARGETS}" PARENT_SCOPE)
set(FINUFFT_FFT_FIND_MODULE "${FINUFFT_FFT_FIND_MODULE}" PARENT_SCOPE)
