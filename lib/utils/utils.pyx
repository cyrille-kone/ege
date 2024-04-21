# distutils: language = c++
# distutils: extra_compile_args=["-std=c++17"]
# coding=utf-8
r"""
PyCharm Editor
Author @git cyrille-kone
"""

cimport  cython
import numpy as np
cimport numpy as np
from libc.math cimport fmax, fmin
from lib.policies cimport Policy
from lib.bandits cimport Bandit
#from libcpp.vector cimport vector
from libcpp.vector cimport vector
np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def pareto_optimal_set(np.ndarray[double, ndim=2, mode="c"] data, double eps=0.0):
    r"""
    Find the Pareto optimal set
    :param eps:
    :param data:
    :return:
    """
    _shape = data.shape
    #assert len(shape) == 2
    cdef size_t dim1, dim2
    dim1 = _shape[0]
    dim2 = _shape[1]
    # require contiguous array
    #cdef np.ndarray[np.double_t, ndim=2, mode = 'c'] np_buff = np.ascontiguousarray(data, dtype=np.uint32)
    #cdef unsigned int * im_buff = <unsigned int *> np_buff.data
    data_array_input = <double*> data.data
    out_array = <np.ndarray> np.zeros(dim1, dtype=np.uint8)
    out_array_data = <np.uint8_t*> np.PyArray_DATA(out_array)
    #with nogil:
    pareto_optimal_arms(dim1, dim2, data_array_input, eps, out_array_data)
    return out_array.astype(bool)

@cython.boundscheck(False)
@cython.wraparound(False)
def minimum_quantity_dom_func(np.ndarray xi, np.ndarray xj, double eps=0.)-> np.double:
    r"""
    The smallest quantity to add uniformly to xj such that it weakly
    dominates xi
    @reference Pareto gap M_eps in Auer et al. 2016 Pareto front identification with stochastic feedback"""
    cdef size_t k, dim
    cdef double res = -np.inf
    dim = xi.shape[0]
    for k in range(dim):
        res = fmax(res, xi[k] + eps - xj[k])
    return fmax(0., res)

@cython.boundscheck(False)
@cython.wraparound(False)
def minimum_quantity_non_dom_func_b(double[:,:] xi, double[:,:] xj, double eps=0.) -> np.double:
    r"""
    The smallest quantity to add uniformly to xi such that it is not strictly
    dominated by xj
    Pareto gap m in Auer et al. 2016 Pareto front identification with stochastic feedback"""
    cdef size_t dim1, dim2, i, j
    dim1, dim2 = xi.shape[0], xj.shape[0]
    cdef np.ndarray out_array = np.empty((dim1, dim2), dtype=np.double)
    for i in range(dim1):
        for j in range(dim2):
            out_array[i, j] = minimum_quantity_non_dom_func(xi[i], xj[j], eps)
    return out_array


@cython.boundscheck(False)
@cython.wraparound(False)
def minimum_quantity_non_dom_func(double[:] xi, double[:] xj, double eps=0.) -> np.double:
    r"""
    The smallest quantity to add uniformly to xi such that it is not strictly
    dominated by xj
    Pareto gap m in Auer et al. 2016 Pareto front identification with stochastic feedback"""
    cdef size_t k, dim
    cdef double res = np.inf
    dim = xi.shape[0]
    for k in range(dim):
        res = fmin(res, xj[k] - xi[k] - eps)
    return fmax(0., res)


@cython.boundscheck(False)
@cython.wraparound(False)
def auer_pareto_regret_non_optimal(double[:] xi, double[:,:] x_s)->np.double:
    r"""
    Pareto regret for non-optimal arms as suggested in
    @reference Auer et al. 2016 Pareto front identification with stochastic feedback"""
    cdef size_t j, dim
    cdef double _max = -np.inf
    dim = x_s.shape[0]
    for j in range(dim):
        _max = fmax(_max, minimum_quantity_non_dom_func(xi, x_s[j]))
    return _max #max([minimum_quantity_non_dom_func(xi, xj) for xj in x_s])

@cython.boundscheck(False)
@cython.wraparound(False)
def auer_pareto_regret_non_optimal2(np.ndarray[double,mode="c"] xi, np.ndarray[double,ndim=2, mode="c"] x_s)->np.double:
    r"""
    Pareto regret for non-optimal arms as suggested in
    @reference Auer et al. 2016 Pareto front identification with stochastic feedback"""
    cdef size_t j, dim1, dim2
    cdef double _max = -np.inf
    dim1 = x_s.shape[0]
    dim2 = x_s.shape[1]
    for j in range(dim1):
        _max = fmax(_max, minimum_quantity_non_dom_func_cpp(dim2, &xi[0], &x_s[j,0]))
    return _max

@cython.boundscheck(False)
@cython.wraparound(False)
def auer_pareto_regret_non_optimal3(np.ndarray[double, mode="c"] xi, np.ndarray[double,ndim=2, mode="c"] x_s)-> double:
    r"""
    Pareto regret for non-optimal arms as suggested in
    @reference Auer et al. 2016 Pareto front identification with stochastic feedback"""
    cdef size_t dim1, dim2
    dim1 = x_s.shape[0]
    dim2 = x_s.shape[1]
    cdef double res
    #with nogil:
    res = auer_pareto_regret_non_optimal_c(dim1, dim2, &xi[0], &x_s[0,0])
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def batch_multivariate_normal(np.ndarray[double, ndim=2] batch_mean, np.ndarray[double, ndim=1] cov) -> np.ndarray:
    r"""Batch samples from a multivariate normal
    Parameters
    ----------
    batch_mean: np.ndarray of shape [N, d]
                Batch of multivariate normal means
    cov: np.ndarray of shape [d, d]
                Common covariance matrix
    Returns
    -------
    Samples from N(batch_mean, batch_cov)"""
    cdef size_t batch_size = batch_mean.shape[0]
    cdef size_t k
    cdef double[:,:] samples = np.empty_like(batch_mean)
    for k in range(batch_size):
        samples[k] = np.random.multivariate_normal(mean=batch_mean[k], cov=cov)
    return samples
