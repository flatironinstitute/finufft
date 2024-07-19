CPMAddPackage(
        NAME asan
        GIT_REPOSITORY "https://github.com/arsenm/sanitizers-cmake.git"
        GIT_TAG "master"
        EXCLUDE_FROM_ALL YES
        GIT_SHALLOW YES
        DOWNLOAD_ONLY YES
)

list(APPEND CMAKE_MODULE_PATH ${asan_SOURCE_DIR}/cmake)
