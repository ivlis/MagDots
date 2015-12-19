#!/usr/bin/env python3
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

extra_args = ["-fopenmp",
"-Wno-unused-function", "-Wno-maybe-uninitialized", "-Wno-format"]
extra_macros = []
ext_modules = [ Extension("util_pyx.util", 
    ["util_pyx/util.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_args,
    extra_link_args=extra_args,
    define_macros=extra_macros,
    libraries=["gsl", "gslcblas"])]

setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = ext_modules
        )
