#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil

from setuptools import setup, Extension

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# The directory for the finufft source
srcdir = ".."
contribdir = os.path.join(srcdir, "contrib")

# Hack the legendre_rule_fast code to think that it's C++
lrsrc = "legendre_rule_fast.cpp"
shutil.copyfile(os.path.join(contribdir, "legendre_rule_fast.c"), lrsrc)

srcdir = os.path.join(srcdir, "src")
# AHB removed twopispread.cpp since no longer needed...
srcfiles = [
    "cnufftspread.cpp", "utils.cpp", "common.cpp",
    "finufft1d.cpp", "finufft2d.cpp", "finufft3d.cpp",
]
srcfiles = [os.path.join(srcdir, fn) for fn in srcfiles]
srcfiles += [lrsrc]
if not os.path.exists(os.path.join(srcdir, "finufft.h")):
    raise RuntimeError("Couldn't find the finufft source")

srcfiles += [os.path.join("finufft", "interface.cpp")]
ext = Extension("finufft.interface",
                sources=srcfiles,
                # language="c++",
                include_dirs=[srcdir, contribdir])

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__FINUFFT_SETUP__ = True
import finufft  # NOQA
from finufft.build import build_ext  # NOQA

setup(
    name="finufft",
    version=finufft.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/ahbarnett/finufft",
    license="TBD",
    packages=["finufft"],
    install_requires=["numpy", "pybind11"],
    ext_modules=[ext],
    description="Flatiron Institute Nonuniform Fast Fourier Transform",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE",
                       os.path.join(srcdir, "*.h"),
                       os.path.join(contribdir, "*", "*.h")]},
    include_package_data=True,
    cmdclass=dict(build_ext=build_ext),
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=True,
)
