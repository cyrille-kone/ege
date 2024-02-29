# coding=utf-8
r"""
PyCharm Editor
"""
import os
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
#os.environ["CC"] = "gcc-13"
#os.environ["CXX"] = "g++-13"
extra_compile_args = ["-std=c++17", "-O2", "-march=native", "-fopenmp" ] #"-fopenmp"
extra_link_args = ['-Wl,-rpath,/opt/homebrew/Cellar/gcc/13.2.0/lib/gcc/13', "-fopenmp"]#'-fopenmp'
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
