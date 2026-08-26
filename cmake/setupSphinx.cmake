CPMAddPackage(
    NAME
    sphinx_cmake
    GIT_REPOSITORY
    https://github.com/k0ekk0ek/cmake-sphinx.git
    GIT_TAG
    e13c40a
    DOWNLOAD_ONLY
    YES
)

list(APPEND CMAKE_MODULE_PATH ${sphinx_cmake_SOURCE_DIR}/cmake/Modules)

# requires sphinx and the texext extension (see docs/requirements.txt)
find_package(Sphinx COMPONENTS texext)
if(SPHINX_FOUND)
    message(STATUS "Sphinx found")
    sphinx_add_docs(finufft_sphinx BUILDER html SOURCE_DIRECTORY
                    ${FINUFFT_SOURCE_DIR}/docs
    )
    # serve the built HTML at http://localhost:8042 for local checking
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_custom_target(
            web
            COMMAND ${CMAKE_COMMAND} -E echo "Serving docs at http://localhost:8042 (Ctrl-C to stop)"
            COMMAND ${Python3_EXECUTABLE} -m http.server 8042 --directory ${CMAKE_CURRENT_BINARY_DIR}/finufft_sphinx
            DEPENDS finufft_sphinx
            COMMENT "Serving docs at http://localhost:8042 (Ctrl-C to stop)"
            USES_TERMINAL
            VERBATIM
        )
    endif()
else()
    message(
        STATUS
        "Sphinx or its texext extension not found - docs targets disabled. "
        "Set up a docs environment with uv:\n"
        "  uv venv ~/.venvs/finufft-docs   # skip if the venv already exists\n"
        "  uv pip install --python ~/.venvs/finufft-docs/bin/python sphinx "
        "-r docs/requirements.txt   # this alone suffices if the venv exists\n"
        "  source ~/.venvs/finufft-docs/bin/activate   # needed before running cmake\n"
        "or with plain pip:\n"
        "  python3 -m venv ~/.venvs/finufft-docs   # skip if the venv already exists\n"
        "  ~/.venvs/finufft-docs/bin/pip install sphinx -r docs/requirements.txt\n"
        "  source ~/.venvs/finufft-docs/bin/activate   # needed before running cmake"
    )
endif()
