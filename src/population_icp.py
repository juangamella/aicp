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
from utils import all_but

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
        (pooled_coefs, pooled_mse) = pooled_regression(target, s, distributions)
        rejected = is_generalizable(s, target, distributions)
        if rejected:
            accepted.append(s)
            mses.append(pooled_mse)
            S = S.intersection(s)
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

def pooled_regression(i, S, distributions):
    pooled_mse = mixture_mse(i, S, distributions)
    coefs = []
    return (coefs, pooled_mse)

def mixture_mse(i, S, distributions):
    """Return the MSE over the mixture of given Gaussian distributions.
    Assuming they are weighted equally its the average of the MSE for each
    distribution.
    """
    mse = 0
    for dist in distributions:
        mse += dist.mse(i,S)
    return mse/len(distributions)
