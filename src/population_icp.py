# Copyright 2020 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""This module contains the "population setting" implementation of
ICP. It relies on sempler's NormalDistribution class.

TODO  BEFORE PUBLISHING:
  - color output by termcolor is not portable to all OSs, so deactivate it
"""

import numpy as np
import itertools
from termcolor import colored
from functools import reduce
from src import utils
from src import icp

def population_icp(distributions, target, debug=False, selection='all', atol=1e-8, rtol=1e-10, check_covariance=False):
    """Perform ICP over a set of Gaussian Distributions, each
    representing a different environment
    """
    assert len(distributions) > 1
    # Build set of candidates
    p = distributions[0].p
    if isinstance(selection, list):
        base = reduce(lambda union, s: set.union(union, s), selection, set())
        candidates = selection
    else:
        if selection=='markov_blanket':
            mb = markov_blanket(target, distributions[0])
            base = set(mb)
        elif selection=='all':
            base = set(range(p))
            base.remove(target)
        candidates = []
        for set_size in range(p):
            candidates += list(itertools.combinations(base, set_size))
    # Evaluate candidates
    accepted = []
    rejected = []
    mses = []
    S = base
    for s in candidates:
        s = set(s)
        reject = reject_hypothesis(s, target, distributions, atol, rtol, check_covariance)
        (pooled_coefs, pooled_intercept, pooled_mse) = pooled_regression(target, list(s), distributions)
        if reject:
            rejected.append(s)
        else:
            accepted.append(s)
            S = S.intersection(s)
            mses.append(pooled_mse)
        if debug:
            color = "red" if reject else "green"
            coefs_str = np.array_str(np.hstack([pooled_coefs, pooled_intercept]), precision=2)
            set_str = "rejected" if reject else "accepted"
            msg = colored("%s %s" % (s, set_str), color) + " Pooled: %s MSE: %0.4f" % (coefs_str, pooled_mse)
            print(msg)
    result = icp.Result(S, accepted, rejected, mses)
    return result

def reject_hypothesis(S, y, distributions, atol=1e-8, rtol=1e-5, check_covariance = False):
    """A set is generalizable if the conditional distribution Y|Xs remains
    invariant across the observed environments. Because we are dealing
    with normal distributions, if the mean and variance are the same, the
    distributions are the same.
    
    The mean is the same if the regression coefficients are the same.
    """
    S = list(S)
    (coefs, intercept) = distributions[0].regress(y,S)
    cov = distributions[0].conditional(y, S, np.zeros_like(S)).covariance
    rejected = False
    i = 1
    while not rejected and i < len(distributions):
        (new_coefs, new_intercept) = distributions[i].regress(y,S)
        if utils.allclose(new_coefs, coefs, rtol, atol) and utils.allclose(new_intercept, intercept, rtol, atol):
            new_cov = distributions[i].conditional(y, S, np.zeros_like(S)).covariance
            if utils.allclose(cov, new_cov, rtol, atol) or not check_covariance:
                i += 1
            else:
                rejected = True
        else:
            rejected = True
    return rejected

def markov_blanket(i, dist, tol=1e-10):

    """Return the Markov blanket of variable i wrt. the given
    distribution. HOW IT WORKS: In the population setting, the
    regression coefficients of all variables outside the Markov
    Blanket are 0. Taking into account numerical issues (that's what
    tol is for), use this to compute the Markov blanket.
    """
    (coefs, _) = dist.regress(i, utils.all_but(i, dist.p))
    return utils.nonzero(coefs, tol)

def pooled_regression(y, S, distributions):
    """Return the coefficients of the linear regressor that minimize the
    MSE over the mixture distribution of environments
    """
    p = distributions[0].p
    if not S: # If only regressing on intercept
        intercept = 0
        coefs = np.zeros(p)
        for dist in distributions:
            intercept += dist.mean[y]
        intercept = intercept / len(distributions)
        return (coefs, intercept, mixture_mse(y, S, distributions))
    else:
        corr_yxs = np.zeros([len(distributions), len(S)+1])
        corr_xs = np.zeros([len(S)+1, len(S)+1, len(distributions)])
        for i,dist in enumerate(distributions):
            mean_xs = np.hstack([dist.mean[S], 1])
            # Build correlation of Xs term
            corr_xs[0:len(S), 0:len(S), i] = utils.matrix_block(dist.covariance, S, S)
            corr_xs[:,:,i] += np.outer(mean_xs, mean_xs)
            # Build correlation of Y with Xs term
            corr_yxs[i, 0:len(S)] = dist.covariance[y, S]
            corr_yxs[i, :] += dist.mean[y] * mean_xs
        pooled_corr_yxs = np.sum(corr_yxs, axis=0)
        pooled_corr_xs = np.sum(corr_xs, axis=2)
        coefs = np.zeros(p + 1)
        coefs[S + [p]] = pooled_corr_yxs @ np.linalg.inv(pooled_corr_xs)
        mse = mixture_mse(y, S, distributions)
        return (coefs[0:p], coefs[p], mse)

def mixture_mse(i, S, distributions):
    """Return the MSE over the mixture of given Gaussian distributions.
    Assuming they are weighted equally its the average of the MSE for each
    distribution.
    """
    mse = 0
    for dist in distributions:
        mse += dist.mse(i,S)
    return mse/len(distributions)

def stable_blanket(accepted, mses, tol=1e-14):
    """The stable blanket is intervention stable (therefore accepted) and
    regression optimal wrt. the observed environments => the set with
    lower MSE is the stable blanket; if ties pick smallest
    """
    mses = np.array(mses)
    accepted = np.array(accepted)
    optimals = accepted[mses < min(mses) + tol] # Sets with lowest MSE
    lengths = np.array(map(len, accepted))
    return optimals[lengths.argmin()]

