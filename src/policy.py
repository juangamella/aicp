# Copyright 2020 Juan Luis Gamella Martin

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
from src import utils, population_icp
from sklearn import linear_model
import warnings

def ratios(p, accepted):    
    one_hot = np.zeros((len(accepted), p))
    for i,s in enumerate(accepted):
        one_hot[i, list(s)] = 1
    return one_hot.sum(axis=0) / len(accepted)

class Policy():
    """Policy class, inherited by all policies. Defines first
    (first_intervention) and next (next_intervention).
    """
    def __init__(self, target, p, name=None):
        self.target = target
        self.p = p # no. of variables
        self.name = name

    def first(self, observational_data):
        """Returns the initial intervention"""
        return None

    def next(self, icp_results):
        """Returns the next intervention"""
        return None

# --------------------------------------------------------------------
# Population setting policies

class RandomPolicy(Policy):
    """Random policy: selects a previously unintervened variable at
    random. Once all variables have been intervened once, repeats the order
    of interventions.
    """
    def __init__(self, target, p, name):
        self.interventions = []
        Policy.__init__(self, target, p, name)
    
    def first(self, _):
        self.idx = np.random.permutation(utils.all_but(self.target, self.p))
        self.i = 0
        return self.random_intervention()

    def next(self, result):
        return (self.random_intervention(), result.accepted)

    def random_intervention(self):
        var = self.idx[self.i]
        self.i = (self.i+1) % len(self.idx)
        return var

class MBPolicy(Policy):
    """Markov Blanket policy: only considers subsets (and intervenes on
    variables) of the Markov blanket """

    def __init__(self, target, p, name, alpha=0.01):
        self.interventions = []
        self.alpha = alpha # significance level for estimating the MB
        Policy.__init__(self, target, p, name)

    def first(self, e):
        mb = population_icp.markov_blanket(self.target, e)
        if len(mb) == 0: # If the estimate of the MB is empty resort to the random strategy
            self.mb = np.random.permutation(utils.all_but(self.target, self.p))
        else:
            self.mb = np.random.permutation(mb)
        self.i = 0
        return self.pick_intervention()

    def next(self, result):
        return (self.pick_intervention(), result.accepted)
    
    def pick_intervention(self):
        var = self.mb[self.i]
        self.i = (self.i+1) % len(self.mb)
        return var

class RatioPolicy(Policy):

    def __init__(self, target, p, name):
        self.interventions = []
        self.current_ratios = np.zeros(p)
        Policy.__init__(self, target, p, name)

    def first(self, e):
        self.mb = set(population_icp.markov_blanket(self.target, e))
        self.candidates = self.mb.copy()
        if self.mb == set():
            var = np.random.choice(utils.all_but(self.target, self.p))
        else:
            var = np.random.choice(list(self.mb))
        self.interventions.append(var)
        return var

    def next(self, result):
        #print("ratios = %s candidates = %s, interventions = %s" % (self.current_ratios, self.candidates, self.interventions))
        new_ratios = ratios(self.p, result.accepted)
        #print(new_ratios)
        to_remove = []
        for i,r in enumerate(new_ratios):
            if r < 0.5 and i in self.candidates:
                to_remove.append(i)
                self.candidates.remove(i)
        self.current_ratios = new_ratios
        # Filter new selection
        selection = result.accepted
        if set() in result.accepted:
            last_intervention = self.interventions[-1]
            selection = [s for s in selection if last_intervention not in s]
        selection = [s for s in selection if len(set.intersection(s, set(to_remove))) == 0]
        return (self.pick_intervention(), selection)

    def pick_intervention(self):
        choice = set.difference(self.candidates, set(self.interventions))
        if choice == set(): # Have intervened on all variables or there are no parents
            None
        else:
            var = np.random.choice(list(choice))
            self.interventions.append(var)
            return var
    
# --------------------------------------------------------------------
# Finite sample setting policies

def markov_blanket(sample, target, tol=1e-3, debug=False):
    """Use the Lasso estimator to return an estimate of the Markov Blanket"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = sample.shape[1]
        predictors = utils.all_but(target, p)
        X = sample[:, predictors]
        Y = sample[:, target]
        coefs = np.zeros(p)
        coefs[predictors] = linear_model.LassoCV(cv=5, normalize=True, max_iter=1000, verbose=debug).fit(X, Y).coef_
        return utils.nonzero(coefs, tol)

class RandomPolicyF(Policy):
    """Random policy: selects variables at random.
    """
    def __init__(self, target, p, name, speedup):
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, _):
        return self.pick_intervention()

    def next(self, result):
        selection = result.accepted if self.speedup else 'all'
        return (self.pick_intervention(), selection)

    def pick_intervention(self):
        return np.random.choice(utils.all_but(self.target, self.p))

class MarkovPolicyF(Policy):
    """Markov policy: selects variables at random from Markov blanket
    estimate.
    """
    def __init__(self, target, p, name, speedup):
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, sample):
        self.mb = markov_blanket(sample, self.target)
        var = self.pick_intervention()
        return var

    def next(self, result):
        var = self.pick_intervention()
        selection = result.accepted if self.speedup else 'all'
        return (var, selection)
    
    def pick_intervention(self):
        if len(self.mb) == 0:
            return None
        else:
            return np.random.choice(self.mb)

class ProposedPolicyMEF(Policy):
    """Proposed policy 1: selects variables at random from Markov blanket
    estimate, removes variables which accept the empty set

    """    
    def __init__(self, target, p, name, speedup):
        self.interventions = []
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, sample):
        self.candidates = set(markov_blanket(sample, self.target))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result):
        last_intervention = self.interventions[-1]
        to_remove = set()
        # Prune candidate set
        if set() in result.accepted:
            to_remove.add(last_intervention)
        self.candidates.difference_update(to_remove)
        # Prune new candidate sets
        selection = result.accepted if self.speedup else utils.combinations(self.p, self.target)
        selection = [s for s in selection if len(set.intersection(s, to_remove)) == 0]
        # Pick next intervention
        var = self.pick_intervention()
        self.interventions.append(var)
        return (var, selection)
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            return np.random.choice(list(self.candidates))

class ProposedPolicyMRF(Policy):
    """Proposed policy 2: selects variables at random from Markov blanket
    estimate, removes variables which accept the empty set or have
    ratio < 1/2

    """
    def __init__(self, target, p, name, speedup):
        self.interventions = []
        self.current_ratios = np.ones(p) * 0.5
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, sample):
        self.candidates = set(markov_blanket(sample, self.target)) # set(range(self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result):
        self.current_ratios = ratios(self.p, result.accepted)
        # Pick next intervention
        var = self.pick_intervention()
        selection = result.accepted if self.speedup else 'all'
        self.interventions.append(var)
        return (var, selection)
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            below_half = set()
            for i,r in enumerate(self.current_ratios):
                if r < 0.5:
                    below_half.add(i)
            choice = set.difference(self.candidates, below_half)
            if len(choice) == 0:
                return np.random.choice(list(self.candidates))
            else:
                return np.random.choice(list(choice))

        
class ProposedPolicyMERF(Policy):
    """Proposed policy 2: selects variables at random from Markov blanket
    estimate, removes variables which accept the empty set or have
    ratio < 1/2

    """
    def __init__(self, target, p, name, speedup):
        self.interventions = []
        self.current_ratios = np.ones(p) * 0.5
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, sample):
        self.candidates = set(markov_blanket(sample, self.target)) # set(range(self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result):
        self.current_ratios = ratios(self.p, result.accepted)
        last_intervention = self.interventions[-1]
        to_remove = set()
        # Prune candidate set
        if set() in result.accepted:
            to_remove.add(last_intervention)
        self.candidates.difference_update(to_remove)
        # Prune candidate sets
        selection = result.accepted if self.speedup else utils.combinations(self.p, self.target)
        selection = [s for s in selection if len(set.intersection(s, to_remove)) == 0]
        # Pick next intervention
        var = self.pick_intervention()
        self.interventions.append(var)
        return (var, selection)
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            below_half = set()
            for i,r in enumerate(self.current_ratios):
                if r < 0.5:
                    below_half.add(i)
            choice = set.difference(self.candidates, below_half)
            if len(choice) == 0:
                return np.random.choice(list(self.candidates))
            else:
                return np.random.choice(list(choice))

class ProposedPolicyEF(Policy):
    """Proposed policy 1: selects variables at random from Markov blanket
    estimate, removes variables which accept the empty set

    """    
    def __init__(self, target, p, name, speedup):
        self.interventions = []
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, sample):
        self.candidates = set(utils.all_but(self.target, self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result):
        last_intervention = self.interventions[-1]
        to_remove = set()
        # Prune candidate set
        if set() in result.accepted:
            to_remove.add(last_intervention)
        self.candidates.difference_update(to_remove)
        # Prune candidate sets
        selection = result.accepted if self.speedup else utils.combinations(self.p, self.target)
        selection = [s for s in selection if len(set.intersection(s, to_remove)) == 0]
        # Pick next intervention
        var = self.pick_intervention()
        self.interventions.append(var)
        return (var, selection)
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            return np.random.choice(list(self.candidates))
    
class ProposedPolicyERF(Policy):
    """Proposed policy 2: selects variables at random from Markov blanket
    estimate, removes variables which accept the empty set or have
    ratio < 1/2

    """
    def __init__(self, target, p, name, speedup):
        self.interventions = []
        self.current_ratios = np.ones(p) * 0.5
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, sample):
        self.candidates = set(utils.all_but(self.target, self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result):
        self.current_ratios = ratios(self.p, result.accepted)
        last_intervention = self.interventions[-1]
        to_remove = set()
        # Prune candidate set
        if set() in result.accepted:
            to_remove.add(last_intervention)
        self.candidates.difference_update(to_remove)
        # Prune candidate sets
        selection = result.accepted if self.speedup else utils.combinations(self.p, self.target)
        selection = [s for s in selection if len(set.intersection(s, to_remove)) == 0]
        # Pick next intervention
        var = self.pick_intervention()
        self.interventions.append(var)
        return (var, selection)
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            below_half = set()
            for i,r in enumerate(self.current_ratios):
                if r < 0.5:
                    below_half.add(i)
            choice = set.difference(self.candidates, below_half)
            if len(choice) == 0:
                return np.random.choice(list(self.candidates))
            else:
                return np.random.choice(list(choice))

class ProposedPolicyRF(Policy):
    """Proposed policy 2: selects variables at random from Markov blanket
    estimate, removes variables which accept the empty set or have
    ratio < 1/2

    """
    def __init__(self, target, p, name, speedup):
        self.interventions = []
        self.current_ratios = np.ones(p) * 0.5
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, sample):
        self.candidates = set(utils.all_but(self.target, self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result):
        self.current_ratios = ratios(self.p, result.accepted)
        # Pick next intervention
        var = self.pick_intervention()
        selection = result.accepted if self.speedup else 'all'
        self.interventions.append(var)
        return (var, selection)
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            below_half = set()
            for i,r in enumerate(self.current_ratios):
                if r < 0.5:
                    below_half.add(i)
            choice = set.difference(self.candidates, below_half)
            if len(choice) == 0:
                return np.random.choice(list(self.candidates))
            else:
                return np.random.choice(list(choice))

class ProposedPolicyEsoftF(Policy):
    """Proposed policy 1: selects variables at random from Markov blanket
    estimate, removes variables which accept the empty set

    """    
    def __init__(self, target, p, name, speedup):
        self.interventions = []
        self.speedup = speedup
        Policy.__init__(self, target, p, name)
    
    def first(self, sample):
        self.candidates = set(utils.all_but(self.target, self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result):
        last_intervention = self.interventions[-1]
        to_remove = set()
        # Prune candidate set
        if set() in result.accepted:
            to_remove.add(last_intervention)
        self.candidates.difference_update(to_remove)
        # Do not discard sets where last intervention appears
        selection = result.accepted if self.speedup else 'all'
        # Pick next intervention
        var = self.pick_intervention()
        self.interventions.append(var)
        return (var, selection)
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            return np.random.choice(list(self.candidates))
