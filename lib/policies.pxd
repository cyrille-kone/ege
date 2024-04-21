# distutils: language = c++
# coding=utf-8
r"""
PyCharm Editor
Author @git cyrille-kone
"""
import numpy as np
cimport numpy as np
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from .bandits cimport bandit, Bandit
from cython.operator cimport dereference as deref
# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef extern from "../src/cpp/policies.cxx":
    pass
# Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass policy_fb:
        policy_fb() except+;
        policy_fb(bandit& bandit_ref) except+;
        size_t K;
        size_t dim;
        size_t D;
        double sigma;
        vector[size_t] action_space;
        bandit* bandit_ref;
        np.npy_bool loop() nogil;


#Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass ege_sr(policy_fb):
        size_t K, D, dim,
        vector[size_t] action_space;
        bandit * bandit_ref;
        ege_sr() except+;
        ege_sr(bandit &) except+;
        np.npy_bool loop(size_t, size_t, size_t) nogil;

#Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass ege_srk(policy_fb):
        size_t K, D, dim,
        vector[size_t] action_space;
        bandit * bandit_ref;
        ege_srk() except+;
        ege_srk(bandit &) except+;
        pair[np.npy_bool, pair[vector[size_t], pair[size_t, size_t]]] loop(size_t, size_t, size_t) nogil;

#Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass ua(policy_fb):
        size_t K, D, dim,
        vector[size_t] action_space;
        bandit * bandit_ref;
        ua() except+;
        ua(bandit &) except+;
        np.npy_bool loop(size_t, size_t) nogil;


#Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass ege_sh(policy_fb):
        size_t K, D, dim,
        vector[size_t] action_space;
        bandit * bandit_ref;
        ege_sh() except+;
        ege_sh(bandit &) except+;
        np.npy_bool loop(size_t, size_t) nogil;



#Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef vector[double] batch_sr(bandit& bandit_ref, vector[size_t]& Ts, vector[size_t]& seeds,size_t k) ;
    cdef pair[vector[double], pair[vector[vector[vector[size_t]]], pair[vector[double], vector[double]]]] batch_srk(bandit& bandit_ref, vector[size_t]& Ts, vector[size_t]& seeds,size_t k);
    cdef vector[double] batch_ua(bandit& bandit_ref, vector[size_t]& Ts, vector[size_t]& seeds) ;
    cdef vector[double] batch_sh(bandit& bandit_ref, vector[size_t]& Ts, vector[size_t]& seeds);

# Define Python interfaces
cdef class Policy:
    cdef readonly size_t K;
    cdef readonly size_t D;
    cdef readonly size_t dim;
    cdef readonly double sigma;
    cdef Bandit py_bandit;
    cdef bandit* bandit_ref
    #cdef policy* policy_ref;
    cdef readonly vector[size_t] action_space;

cdef class py_ege_sr(Policy):
    cdef ege_sr* policy_ref
cdef class py_ege_srk(Policy):
    cdef ege_srk* policy_ref
cdef class py_ua(Policy):
    cdef ua* policy_ref
cdef class py_ege_sh(Policy):
    cdef ege_sh* policy_ref

#cdef inline py_batch_sr(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds, size_t  k):
#    pass

#cdef inline py_batch_ua(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds):
#    return batch_ua(deref(bandit_py.bandit_ref), Ts, seeds)

