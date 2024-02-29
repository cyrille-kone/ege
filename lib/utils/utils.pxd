# distutils: language = c++
# distutils: extra_compile_args=["-std=c++17"]
# coding=utf-8
r"""
PyCharm Editor
Author @git cyrille-kone
"""


# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cimport numpy as np
# init numpy
np.import_array()
cdef extern from "src/c_utils.cpp":
    pass

# Declare the class with cdef
cdef extern from "src/c_utils.h":
        cdef void pareto_optimal_arms(const size_t, const size_t, double*, const double, np.uint8_t*) nogil
        cdef double minimum_quantity_non_dom_func_cpp(const size_t, double *, double *);
        cdef double auer_pareto_regret_non_optimal_c(const size_t, const size_t, double *, double *) nogil;
        cdef double minimum_quantity_dom_func_cpp(const size_t, double*, double*);
        cdef void arms_dominating_mu_i(const size_t, const size_t, double*, double*, double, unsigned char*);