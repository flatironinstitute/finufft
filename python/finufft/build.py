# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import tempfile

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

__all__ = ["build_ext"]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError("Unsupported compiler -- at least C++11 support "
                           "is needed!")

class build_ext(_build_ext):
    """
    A custom extension builder that finds the include directories for Eigen
    before compiling.

    """

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        # Add the libraries
        libraries = ["fftw3", "fftw3_threads"]
        if os.name == "posix":
            libraries += ["m", "stdc++"]
        for ext in self.extensions:
            ext.libraries += libraries

        # Add the numpy and pybind11 include directories
        import numpy
        import pybind11
        include_dirs = [
            numpy.get_include(),
            pybind11.get_include(False),
            pybind11.get_include(True),
        ]
        for ext in self.extensions:
            ext.include_dirs += include_dirs

        # Set up pybind11
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append("-DVERSION_INFO=\"{0:s}\""
                        .format(self.distribution.get_version()))
            opts.append(cpp_flag(self.compiler))
            for flag in ["-funroll-loops", "-fvisibility=hidden",
                         "-Wno-unused-function", "-Wno-uninitialized", "-O4"]:
                if has_flag(self.compiler, flag):
                    opts.append(flag)
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"{0:s}\\"'
                        .format(self.distribution.get_version()))
        for ext in self.extensions:
            ext.extra_compile_args = opts

        # Run the standard build procedure.
        _build_ext.build_extensions(self)
