# Overlay port: it builds the FINUFFT checkout that holds this file. CPM downloads
# xsimd, POET, findFFTW and DUCC0 at build time, so the install needs network.
# vcpkg's ABI hash covers this port directory, not the checkout, so after a change
# to the checkout reinstall with the binary cache off (--binarysource=clear).
get_filename_component(SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(NOT "fftw" IN_LIST FEATURES AND NOT "ducc0" IN_LIST FEATURES)
    message(FATAL_ERROR "finufft needs an FFT: enable the fftw or the ducc0 feature")
endif()

vcpkg_check_features(
    OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        ducc0 FINUFFT_USE_DUCC0
        openmp FINUFFT_USE_OPENMP
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" FINUFFT_STATIC)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DFINUFFT_STATIC_LINKING=${FINUFFT_STATIC}
        -DFINUFFT_BUILD_TESTS=OFF
        -DFINUFFT_BUILD_EXAMPLES=OFF
        -DFINUFFT_BUILD_PYTHON=OFF
        -DFINUFFT_ENABLE_INSTALL=ON
        -DFINUFFT_FFTW_LIBRARIES=DEFAULT
        # CPM downloads through FetchContent; vcpkg disconnects FetchContent
        # by default, so turn it back on.
        -DFETCHCONTENT_FULLY_DISCONNECTED=OFF
    MAYBE_UNUSED_VARIABLES
        FINUFFT_FFTW_LIBRARIES
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME finufft CONFIG_PATH lib/cmake/finufft)
vcpkg_fixup_pkgconfig()
vcpkg_copy_pdbs()

file(
    REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
    "${CURRENT_PACKAGES_DIR}/share/finufft/examples"
    "${CURRENT_PACKAGES_DIR}/share/licenses"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
