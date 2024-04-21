# distutils: language = c++
# coding=utf-8
r"""
PyCharm Editor
Author @git cyrille-kone
"""
cimport cython
import numpy as np
from .bandits import Bandit
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref


cdef class Policy(object):
    def __init__(self, Bandit py_bandit):
        self.K = py_bandit.K
        self.D = py_bandit.D
        self.sigma = py_bandit.sigma
        self.dim = py_bandit.D
        self.action_space = py_bandit.action_space
        self.py_bandit = py_bandit


cdef class py_ege_sr(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ege_sr(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T, size_t k):
        return self.policy_ref.loop(seed, T, k)

cdef class py_ege_srk(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ege_srk(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T, size_t k):
        return self.policy_ref.loop(seed, T, k)

cdef class py_ege_sh(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ege_sh(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T):
        return self.policy_ref.loop(seed, T)


cdef class py_ua(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ua(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T):
        return self.policy_ref.loop(seed, T)

cpdef py_batch_sr(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds, size_t  k):
    return batch_sr(deref(bandit_py.bandit_ref), Ts, seeds, k)

cpdef py_batch_srk(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds, size_t  k):
    return batch_srk(deref(bandit_py.bandit_ref), Ts, seeds, k)

cpdef py_batch_ua(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds):
    return batch_ua(deref(bandit_py.bandit_ref), Ts, seeds)

cpdef py_batch_sh(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds):
    return batch_sh(deref(bandit_py.bandit_ref), Ts, seeds)