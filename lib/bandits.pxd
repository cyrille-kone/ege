# distutils: language = c++
# coding=utf-8
r"""
PyCharm Editor
Author @git cyrille-kone
"""
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef extern from "../src/cpp/bandits.cxx":
    pass
# Declare the class with cdef
cdef extern from "../src/cpp/bandits.hpp":
    cdef cppclass bandit:
     bandit() except+;
     bandit(vector[vector[double]]& arms_means) except+;
     vector[vector[double]] sample(const vector[size_t]& arms);
     void reset_env(size_t seed);
     vector[size_t] pareto_optimal_arms;
     vector[double] suboptimal_gaps;
     vector[size_t] optimal_arms;
     double H;
     size_t K;
     size_t D;
     size_t seed;
     double sigma;
     vector[size_t] action_space;
     size_t optimal_arm; #no always defined;


cdef class Bandit:
 cdef readonly size_t K;
 cdef readonly size_t D;
 cdef readonly double sigma;
 cdef readonly vector[double] stddev;
 cdef readonly size_t seed;
 cdef readonly vector[size_t] action_space;
 cdef readonly vector[double] suboptimal_gaps;
 cdef readonly double H;
 cdef bandit* bandit_ref;
 cdef readonly vector[size_t] optimal_arms;
 #cpdef np.ndarray[double, ndim=2] sample(self, vector[size_t])

cdef class Bernoulli(Bandit):
 pass
# bernoulli bandit
# Declare the class with cdef
cdef extern from "../src/cpp/bandits.hpp":
    cdef cppclass bernoulli(bandit):
     bernoulli() except+;
     bernoulli(vector[vector[double]]& arms_means) except+;
     vector[vector[double]] sample(const vector[size_t]& arms);
     void reset_env(size_t seed);
     vector[size_t] optimal_arms;
     vector[double] suboptimal_gaps;
     double H;
     size_t K;
     size_t D;
     size_t seed;
     double sigma;
     vector[size_t] action_space;
     size_t optimal_arm; #not always defined;

cdef class Gaussian(Bandit):
 pass

# Gaussian bandit
# Declare the class with cdef
cdef extern from "../src/cpp/bandits.hpp":
    cdef cppclass gaussian(bandit):
     gaussian() except+;
     gaussian(vector[vector[double]]& arms_means, vector[double]& stddev) except+;
     vector[vector[double]] sample(const vector[size_t]& arms);
     void reset_env(size_t seed);
     vector[size_t] optimal_arms;
     vector[double] suboptimal_gaps;
     vector[double] stddev
     double H;
     size_t K;
     size_t D;
     size_t seed;
     double sigma;
     vector[size_t] action_space;
     size_t optimal_arm; #not always defined;