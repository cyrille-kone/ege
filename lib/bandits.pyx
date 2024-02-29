# distutils: language = c++
# coding=utf-8
r"""
PyCharm Editor
Author @git cyrille-kone
"""
import numpy as np
from libcpp.vector cimport vector
cdef class Bandit(object):
    def __init__(self, vector[vector[double]]& mus):
        self.K = self.bandit_ref.K
        self.D = self.bandit_ref.D
        self.action_space = self.bandit_ref.action_space
        self.sigma = self.bandit_ref.sigma
        self.seed = self.bandit_ref.seed
        self.H = self.bandit_ref.H
        self.suboptimal_gaps = self.bandit_ref.suboptimal_gaps
        self.optimal_arms = self.bandit_ref.optimal_arms
    def sample(self, vector[size_t] arms):
        return self.bandit_ref.sample(arms)
    def reset_env(self, size_t seed=42):
        self.bandit_ref.reset_env(seed)
        return

cdef class Bernoulli(Bandit):
    def __init__(self, vector[vector[double]]& mus):
        self.sigma = 0.5
        super().__init__(mus)
    def __cinit__(self, vector[vector[double]]& mus):
        self.bandit_ref = new bernoulli(mus)

cdef class Gaussian(Bandit):
    def __init__(self, vector[vector[double]]& mus, vector[double]& stddev):
        super().__init__(mus)
        self.stddev = stddev
    def __cinit__(self, vector[vector[double]]& mus, vector[double]& stddev):
        self.bandit_ref = new gaussian(mus, stddev)