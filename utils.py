# @title Install
# %load_ext cython
import abc
import numpy as np
from joblib import Parallel, delayed
inf = (1<<31)*1.
def is_non_dominated(Y: np.ndarray, eps=0.) -> np.ndarray:
    r"""Computes the non-dominated front.
  @ Copyright: this function is modified from boTorch utils
    Note: this assumes maximization.

    For small `n`, this method uses a highly parallel methodology
    that compares all pairs of points in Y. However, this is memory
    intensive and slow for large `n`. For large `n` (or if Y is larger
    than 5MB), this method will dispatch to a loop-based approach
    that is faster and has a lower memory footprint.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    #n = Y.shape[-2]
    Y1 = np.expand_dims(Y, -3)
    Y2 = np.expand_dims(Y, -2)
    # eps from context
    dominates = (Y1  >= Y2 + eps).all(axis=-1) & (Y1 > Y2 + eps).any(axis=-1)
    nd_mask = ~(dominates.any(axis=-1))
    return nd_mask


# @title  Set up
def batch_multivariate_normal(batch_mean, batch_cov) -> np.ndarray:
    r"""Batch samples from a multivariate normal
    Parameters
    ----------
    batch_mean: np.ndarray of shape [N, d]
                Batch of multivariate normal means
    batch_cov: np.ndarray of shape [N, d, d]
                Batch of multivariate normal covariances
    Returns
    -------
    Samples from N(batch_mean, batch_cov)"""
    batch_size = np.shape(batch_mean)[0]
    samples = np.arange(batch_size).astype(np.float32).reshape(-1, 1)
    return np.apply_along_axis(
        lambda i: np.random.multivariate_normal(mean=batch_mean[int(i[0])], cov=batch_cov),
        axis=1,
        arr=samples)


class Bandit(object):
    r"""Base class for bandit sampler"""

    def __init__(self, arms_means):
        self.arms_means = arms_means
        self.K = len(arms_means)
        self.arms_space = range(self.K)

    @abc.abstractmethod
    def sample(self, arms):
        r"""Get batch samples form arms"""
        raise NotImplementedError

    def initialize(self):
        r""" Re-initialize the bandit environment"""


class GaussianBandit(Bandit):
    r"""Implement a Gaussian bandit"""

    def __init__(self, K=2, arms_means=None, arms_scale=None, D=1) -> None:
        r"""
        @constructor
        Parameters
        ----------
        K: int > 0
           Number of arms of the bandit
        arms_means: np.ndarray of shape [K, d]
           Mean reward of each arm
        arms_scale: float or np.ndarray
                   scale or covariance matrix of each arm
        D: int>0
           Dimension of the reward vector
        """
        self._D = D if arms_means is None else [*np.shape(arms_means), 1][1]
        self._arms_means = np.random.uniform(size=(K, self._D)) if arms_means is None else np.asarray(arms_means)
        self._arms_means = self._arms_means.reshape(-1, self._D).squeeze(-1) if self._D == 1 else self._arms_means
        self._arms_scale = np.eye(self._D) if arms_scale is None else arms_scale
        super(GaussianBandit, self).__init__(self._arms_means)

    def sample(self, arms):
        r"""
        Sample from a Gaussiant bandit
        Parameters
        -----------
        arms : set  of arms to sample
        Returns
        ------
        Samples from arms
        Test
        ----
        >>> gaussian_bandit= GaussianBandit(K=10)
        >>> gaussian_bandit.sample([1,2,4])
        """
        arms = [arms] if isinstance(arms, int) else np.asarray(arms, dtype=int)
        if self._D > 1:
            return np.concatenate(
                [np.random.normal(loc=self._arms_means[:, i][arms], scale=np.sqrt(self._arms_scale[i][i])).reshape(-1, 1) for i in range(self._D)], -1)
   
        
            #return batch_multivariate_normal(self._arms_means[arms], self._arms_scale) / np.sqrt(np.diag(self._arms_scale))
        elif self._D == 1:
            return np.random.normal(loc=self._arms_means[arms], scale=self._arms_scale).reshape(-1, 1)
        raise ValueError(f"Value of D should be larger than or equal to 1 but given {self._D}")


class BernoulliBandit(Bandit):
    r"""Implement a Bernoulli bandit"""

    def __init__(self, K=2, arms_means=None, D=1) -> None:
        r"""
        @constructor
        Parameters
        ----------
        K: int > 0
           Number of arms of the bandit
        arms_means: np.ndarray of shape [K, d]
           Mean reward of each arm
        D: int>0
           Dimension of the reward vector
                """
        self._D = D if arms_means is None else [*np.shape(arms_means), 1][1]
        self._arms_means = np.random.uniform(size=(K, self._D)) if arms_means is None else np.asarray(arms_means)
        self._arms_means = self._arms_means.reshape(-1, self._D).squeeze(-1) if self._D == 1 else self._arms_means
        super(BernoulliBandit, self).__init__(self._arms_means)

    def sample(self, arms):
        r"""
         Sample from a Bernoulli bandit
         Parameters
         -----------
         arms : set  of arms to sample
         Returns
         ------
         Samples from arms
         Test
         ----
         >>> bernoulli_bandit = BernoulliBandit(K=10)
         >>> bernoulli_bandit.sample([1,2,4])
         """
        arms = [arms] if isinstance(arms, int) else arms
        return np.random.binomial(1, self.arms_means[arms]).reshape(-1, self._D)


def M(xi, xj):
    r''' @func utils cf paper'''
    return np.max(xi - xj, -1)

def m(xi, xj):
    r''' @func utils cf paper'''
    return np.min(xj - xi, -1)


def delta_i_plus(i, S_star, means):
    r''' gap optimal arm '''
    return min([min(M(means[i], means[j]), M(means[j], means[i])) + inf*(j==i) for j in S_star])


def delta_i_minus(i, S_star_comp, means):
    r""" gap optimal arm """
    if len(S_star_comp) == 0: return inf
    return min([max(M(means[j],means[i]),0) + max(Delta_i_star(j, means),0) for j in S_star_comp])

def delta_i_minus_prime(i, S_star_comp, means):
    if len(S_star_comp) == 0: return inf
    return min([m(means[j],means[i]) for j in S_star_comp if (m(means[j],means[i]) == Delta_i_star(j, means)) ]+[inf]) 

def Delta_i_star(i, means):
    r""" gap sub-optimal arm"""
    return np.max(m(means[i], means))


def Delta_i(i, S_star, S_star_comp, means):
    r""" compute the gap of an optimal arm"""
    if i in S_star: return min(delta_i_plus(i, S_star,means), delta_i_minus(i,S_star_comp,means))
    return Delta_i_star(i, means)

def Delta_i_prime(i, S_star, S_star_comp, means):
    if i in S_star: return min(delta_i_plus(i, S_star,means), delta_i_minus_prime(i,S_star_comp,means))
    return Delta_i_star(i, means)


# @title Utils for the algorithms to run in Parallel
def run_batch_seeds(T, seeds, callback):
    r"""
    Runs callback with a budget T with the seeds in [seeds]
    :param T: Budget
    :param seeds: list of seeds
    :param callback: algorithm to run [ege_sr | ege_sh| round_robin]
    :return: The average return over thr seeds in [seeds]
    """
    return np.mean(Parallel(n_jobs=-1, verbose=0)(delayed(callback)(seed, T) for seed in seeds))



