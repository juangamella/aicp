# Copyright 2019 Juan Luis Gamella Martin

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

import numpy as np
import itertools
from termcolor import colored
from src.utils import all_but, matrix_block

def population_icp(distributions, target, debug=False, selection='all'):
    """Perform ICP over a set of Gaussian Distributions, each
    representing a different environment
    """
    p = distributions[0].p
    base.remove(target)
    if selection=='markov_blanket':
        mb = markov_blanket(target, distributions[0])
        base = set(mb)
    elif selection=='all':
        base = set(range(p))
        base.remove(target)
    S = base.copy()
    accepted = []
    mses = []
    candidates = itertools.combinations(base)
    for s in candidates:
        rejected = is_generalizable(s, target, distributions)
        (coefs, intercept, pooled_mse) = pooled_regression(target, s, distributions)
        if not rejected:
            accepted.append(s)
            S = S.intersection(s)
            mses.append(pooled_mse)
        if debug:
            color = "red" if rejected else "green"
            coefs_str = np.array_str(pooled_coefs, precision=2)
            set_str = "rejected" if rejected else "accepted"
            msg = colored("%s %s" % (s, set_str), color) + " - %s MSE: %0.4f" % (coef_str, error)
    return (S, accepted, mses)

def markov_blanket(i, dist, tol=1e-10):
    """Return the Markov blanket of variable i wrt. the given
    distribution. HOW IT WORKS: In the population setting, the
    regression coefficients of all variables outside the Markov
    Blanket are 0. Taking into account numerical issues (that's what
    tol is for), use this to compute the Markov blanket.
    """
    (coefs, _) = dist.regress(i, all_but(i, dist.p))
    (mb,) = np.where(np.abs(coefs) > tol)
    return mb

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
            corr_xs[0:len(S), 0:len(S), i] = matrix_block(dist.covariance, S, S)
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
