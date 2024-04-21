# coding=utf-8
r"""
PyCharm Editor
"""
import os
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
r"""
Define your own cpp compiler 
If the compiler does not support 'openmp' it should be removed from the args
"""
# Change to use your own compiler otherwise the default compiler is used
# os.environ["CC"] = "gcc-10" # change accordingly
# os.environ["CXX"] = "g++-10" # change accordingly
extra_compile_args = ["-std=c++17", "-O2"]  # add if supported "-fopenmp"
r'''
If the linker does not find the path to the lib of the compiler lib 
Please add it 
e.g for MacOs '-Wl,-rpath,/opt/homebrew/Cellar/gcc/{version}/lib/gcc/{version}'
'''
# add path to the
extra_link_args = [""]  #add if supported '-fopenmp'
language = "c++"
setup(name="aistats23",
      ext_modules=cythonize([
          Extension(
              name="lib.bandits",
              sources=["lib/bandits.pyx",
                       "src/cpp/utils.cxx"],
              language=language,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
          ),
          Extension(
              name="lib.policies",
              sources=["lib/policies.pyx",
                       "src/cpp/utils.cxx"],
              language=language,
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
          )
      ]),
      language_level=3,
      include_dirs=[np.get_include()]
      )
