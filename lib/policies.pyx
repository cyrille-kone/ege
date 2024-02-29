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
        #print(seed, T)
        #print("new print")
        return self.policy_ref.loop(seed, T, k)

cdef class py_ege_srk(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ege_srk(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T, size_t k):
        #print(seed, T)
        #print("new print")
        return self.policy_ref.loop(seed, T, k)

cdef class py_ege_sh(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ege_sh(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T):
        #print(seed, T)
        #print("new print")
        return self.policy_ref.loop(seed, T)


cdef class py_ua(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ua(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T):
        #print(seed, T)
        #print("new print")
        return self.policy_ref.loop(seed, T)

cdef class py_ape_fb(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new psi_ape_fb(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T, double delta=0.1):
        #print(seed, T)
        #print("new print")
        return self.policy_ref.loop(seed, T, delta)

cdef class py_psi_ucbe(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new psi_ucbe(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T, double a):
        return self.policy_ref.loop(seed, T, a)

cdef class py_ape_b(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ape_b(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T, double a):
        return self.policy_ref.loop(seed, T, a)

cdef class py_psi_ucbe_adapt(py_psi_ucbe):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new psi_ucbe_adapt(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T, double c):
        return self.policy_ref.loop(seed, T, c)

cdef class py_ape_b_adapt(py_ape_b):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new ape_b_adapt(deref(self.bandit_ref))
    def loop(self, size_t seed, size_t T, double c):
        return self.policy_ref.loop(seed, T, c)

cpdef py_batch_sr(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds, size_t  k):
    return batch_sr(deref(bandit_py.bandit_ref), Ts, seeds, k)

cpdef py_batch_srk(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds, size_t  k):
    return batch_srk(deref(bandit_py.bandit_ref), Ts, seeds, k)

cpdef py_batch_ua(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds):
    return batch_ua(deref(bandit_py.bandit_ref), Ts, seeds)

cpdef py_batch_sh(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds):
    return batch_sh(deref(bandit_py.bandit_ref), Ts, seeds)

cpdef py_batch_ape(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds, double c):
    return batch_ape(deref(bandit_py.bandit_ref), Ts, seeds, c)

cpdef py_batch_ape_adapt(Bandit bandit_py, vector[size_t]& Ts, vector[size_t]&seeds, double c):
    return batch_ape_adapt(deref(bandit_py.bandit_ref), Ts, seeds, c)