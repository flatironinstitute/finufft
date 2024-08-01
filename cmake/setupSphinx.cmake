cpmaddpackage(
  NAME
  sphinx_cmake
  GIT_REPOSITORY
  https://github.com/k0ekk0ek/cmake-sphinx.git
  GIT_TAG
  e13c40a
  DOWNLOAD_ONLY
  YES)

list(APPEND CMAKE_MODULE_PATH ${sphinx_cmake_SOURCE_DIR}/cmake/Modules)

# requires pip install sphinx texext
find_package(Sphinx)
if(SPHINX_FOUND)
  message(STATUS "Sphinx found")
  sphinx_add_docs(finufft_sphinx BUILDER html SOURCE_DIRECTORY
                  ${FINUFFT_SOURCE_DIR}/docs)
else()
  message(STATUS "Sphinx not found docs will not be generated")
endif()
