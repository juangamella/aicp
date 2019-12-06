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
from src import utils

def population_icp(distributions, target, debug=False, selection='all', test='coefficients'):
    """Perform ICP over a set of Gaussian Distributions, each
    representing a different environment
    """
    p = distributions[0].p
    if selection=='markov_blanket':
        mb = markov_blanket(target, distributions[0])
        base = set(mb)
    elif selection=='all':
        base = set(range(p))
        base.remove(target)
    S = base.copy()
    accepted = []
    mses = []
    all_sets = []
    for set_size in range(p):
        candidates = itertools.combinations(base, set_size)
        for s in candidates:
            s = set(s)
            all_sets.append(s)
            rejected = reject_hypothesis(s, target, distributions)
            (pooled_coefs, pooled_intercept, pooled_mse) = pooled_regression(target, list(s), distributions)
            if not rejected:
                accepted.append(s)
                S = S.intersection(s)
                mses.append(pooled_mse)
            if debug:
                color = "red" if rejected else "green"
                coefs_str = np.array_str(np.hstack([pooled_coefs, pooled_intercept]), precision=2)
                set_str = "rejected" if rejected else "accepted"
                msg = colored("%s %s" % (s, set_str), color) + " Pooled: %s MSE: %0.4f" % (coefs_str, pooled_mse)
                print(msg)
    return (S, accepted, mses, all_sets)

def reject_hypothesis(S, y, distributions):
    """Sets are stable if their regression coefficients are the same for
    all observed environments. NOTE: The empty set will always
    satisify this requirement as long as the interventions do not
    shift the mean of the target => for the empty set, check
    additionally if the variance of the target has changed.
    """
    S = list(S)
    (coefs, intercept) = distributions[0].regress(y,S)
    rejected = False
    i = 1
    while not rejected and i < len(distributions):
        (new_coefs, new_intercept) = distributions[i].regress(y,S)
        rejected = not np.allclose(coefs, new_coefs) or not np.allclose(intercept, new_intercept)
        if not S and distributions[i].marginal(y).covariance != distributions[i-1].marginal(y).covariance:
            rejected = True
        i += 1
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

# from src import sampling
# from functools import reduce
# y = 3
# #W, ordering, _, _ = utils.eg3()
# W, ordering = sampling.dag_avg_deg(8,2.5,1,1,debug=True,random_state=2)
# sem = sampling.LGSEM(W, ordering, (1,1))
# e = sem.sample(population=True)
# e_obs = sem.sample(n=round(1e6))
# v = 2

# e0 = sem.sample(population=True, noise_interventions=np.array([[0,0,v]]))
# e1 = sem.sample(population=True, noise_interventions=np.array([[1,0,v]]))
# e2 = sem.sample(population=True, noise_interventions=np.array([[2,0,v]]))
# e3 = sem.sample(population=True, noise_interventions=np.array([[3,0,v]]))
# e4 = sem.sample(population=True, noise_interventions=np.array([[4,0,v]]))
# e5 = sem.sample(population=True, noise_interventions=np.array([[5,0,v]]))
# e6 = sem.sample(population=True, noise_interventions=np.array([[6,0,v]]))
# e7 = sem.sample(population=True, noise_interventions=np.array([[7,0,v]]))

# e24 = sem.sample(population=True, noise_interventions=np.array([[2,0.3,v], [4, 0.3, v]]))

# print()
# print(pooled_regression(y, [], [e0]))
# print(pooled_regression(y, [], [e1]))
# print(pooled_regression(y, [], [e2]))
# print(pooled_regression(y, [], [e3]))
# print(pooled_regression(y, [], [e4]))

# print()
# S = []
# s = np.random.uniform(size=len(S))
# print(e0.conditional(y, S, s).mean, e0.conditional(y, S, s).covariance)
# print(e1.conditional(y, S, s).mean, e1.conditional(y, S, s).covariance)
# print(e2.conditional(y, S, s).mean, e2.conditional(y, S, s).covariance)
# print(e3.conditional(y, S, s).mean, e3.conditional(y, S, s).covariance)
# print(e4.conditional(y, S, s).mean, e4.conditional(y, S, s).covariance)

# (S, accepted, mses, all_sets) = population_icp([e], y, debug=True, selection='all')
# print(S, accepted, mses)
# print("Stable blanket :", stable_blanket(accepted, mses))
# print("Nint:", reduce(lambda acc,x: acc.union(x), accepted))

# #utils.plot_graph(W, ordering)
