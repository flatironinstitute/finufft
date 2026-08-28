vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO flatironinstitute/finufft
    REF "v${VERSION}"
    SHA512 a5f782eb16a5317ade277db8b1925472e448602abd0c8b5a24f6433db2452dc116782590bb7e49344e65a3c7528457479403dc588a508edb6c6b5c31e04ce35a
    HEAD_REF master
)

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
        # vcpkg supplies FFTW; never let the build download its own copy.
        -DFINUFFT_FFTW_LIBRARIES=DEFAULT
        # xsimd, poet and the findFFTW module still come from CPM at configure
        # time, and none of the three has a vcpkg port. vcpkg_cmake_configure
        # PREPENDs -DFETCHCONTENT_FULLY_DISCONNECTED=ON, so this later -D wins
        # and the fetches are allowed again. An upstream vcpkg submission needs
        # ports for those three first; until then this port is an overlay.
        -DFETCHCONTENT_FULLY_DISCONNECTED=OFF
    MAYBE_UNUSED_VARIABLES
        FINUFFT_FFTW_LIBRARIES
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME finufft CONFIG_PATH lib/cmake/finufft)
vcpkg_copy_pdbs()

file(
    REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
    "${CURRENT_PACKAGES_DIR}/share/finufft/examples"
    "${CURRENT_PACKAGES_DIR}/share/licenses"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
