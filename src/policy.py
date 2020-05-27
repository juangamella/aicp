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

"""This module contains the intervention selection policies (as classes).

Policies are named after the strategies they employ:
  - Markov: Markov blanket strategy
  - E: empty-set strategy
  - R: ratio strategy

Population policies have a slightly different behaviour,
e.g. variables are only intervened on once, and are
marked with "Pop".

"""

import numpy as np
from src import utils, population_icp, icp
from src.utils import ratios
from sklearn import linear_model
from functools import reduce
import warnings

def intersection(sets):
    """Return the intersection of a collection of sets"""
    if sets == []:
        return set()
    else:
        return reduce(lambda s, acc: acc & s, sets)

class Policy():
    """Policy class, inherited by all policies. Defines first
    (first_intervention) and next (next_intervention).
    """
    def __init__(self, target, p, name, alpha=None):
        self.target = target
        self.p = p # no. of variables
        self.name = name
        self.alpha = alpha
        self.interventions = []

    def first(self, observational_sample):
        """Returns the initial intervention"""
        return None

    def next(self, icp_results, interventional_sample):
        """Returns the next intervention"""
        return None

# --------------------------------------------------------------------
# Population setting policies

class PopRandom(Policy):
    """Random policy: selects a previously unintervened variable at
    random. Once all variables have been intervened once, repeats the order
    of interventions.
    """
    def __init__(self, target, p, name):
        Policy.__init__(self, target, p, name)
        
    def first(self, _):
        self.idx = np.random.permutation(utils.all_but(self.target, self.p))
        self.i = 0
        return self.random_intervention()

    def next(self, result, _):
        return self.random_intervention()

    def random_intervention(self):
        var = self.idx[self.i]
        self.i = (self.i+1) % len(self.idx)
        return var

class PopMarkov(Policy):
    """Markov Blanket policy: selects a previously unintervened variable
    from the Markov blanket"""

    def __init__(self, target, p, name):
        Policy.__init__(self, target, p, name)

    def first(self, e):
        mb = population_icp.markov_blanket(self.target, e)
        if len(mb) == 0: # If the estimate of the MB is empty resort to the random strategy
            self.mb = np.random.permutation(utils.all_but(self.target, self.p))
        else:
            self.mb = np.random.permutation(mb)
            self.i = 0
        return self.pick_intervention()

    def next(self, result, _):
        return self.pick_intervention()
    
    def pick_intervention(self):
        var = self.mb[self.i]
        self.i = (self.i+1) % len(self.mb)
        return var

class PopMarkovR(Policy):
    """Selects a previously unintervened variable from those in the Markov
    blanket which have a ratio above 1/2"""
    def __init__(self, target, p, name):
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

    def next(self, result, _):
        # Ratio strategy
        new_ratios = ratios(self.p, result.accepted)
        for i,r in enumerate(new_ratios):
            if r < 0.5 and i in self.candidates:
                self.candidates.remove(i)
        return self.pick_intervention()

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
        coefs[predictors] = linear_model.LassoCV(cv=10, normalize=True, max_iter=1000, verbose=debug).fit(X, Y).coef_
        return utils.nonzero(coefs, tol)

class Random(Policy):
    """Random policy: selects variables at random.
    """
    def __init__(self, target, p, name, alpha):
        Policy.__init__(self, target, p, name, alpha)
        
    def first(self, _):
        return self.pick_intervention()

    def next(self, result, _):
        return self.pick_intervention()

    def pick_intervention(self):
        return np.random.choice(utils.all_but(self.target, self.p))
 
class E(Policy):
    """empty-set: selects variables at random, removing variables for
    which, after being intervened, the empty set is accepted
    """
    def __init__(self, target, p, name, alpha):
        Policy.__init__(self, target, p, name, alpha)
        
    def first(self, sample):
        self.obs_sample = sample
        self.candidates = set(utils.all_but(self.target, self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result, intervention_sample):
        # Remove identified parents
        self.candidates.difference_update(intersection(result.accepted))
        # Empty-set strategy 2.0
        last_intervention = self.interventions[-1]
        if set() in icp.icp([self.obs_sample, intervention_sample], self.target, self.alpha, selection=[set()]).accepted:
            self.candidates.difference_update({last_intervention})
        # Pick next intervention
        var = self.pick_intervention()
        self.interventions.append(var)
        return var
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            return np.random.choice(list(self.candidates))
        
class R(Policy):
    """ratio: selects variables with a stability ratio above 1/2."""
    def __init__(self, target, p, name, alpha):
        self.current_ratios = np.ones(p) * 0.5
        self.current_ratios[target] = 0
        Policy.__init__(self, target, p, name, alpha)
        
    def first(self, sample):
        self.candidates = set(utils.all_but(self.target, self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result, _):
        # Remove identified parents
        self.candidates.difference_update(intersection(result.accepted))
        # Pick next intervention
        self.current_ratios = ratios(self.p, result.accepted)
        var = self.pick_intervention()
        self.interventions.append(var)
        return var
    
    def pick_intervention(self):
        below_half = set()
        for i,r in enumerate(self.current_ratios):
            if r < 0.5:
                below_half.add(i)
        choice = set.difference(self.candidates, below_half)
        if len(choice) == 0:
            None
        else:
            return np.random.choice(list(choice))
        
class ER(Policy):
    """empty-set + ratio: selects variables with a stability ratio above
    1/2, removing variables for which, after being intervened, the
    empty set is accepted
    """
    def __init__(self, target, p, name, alpha):
        self.current_ratios = np.ones(p) * 0.5
        self.current_ratios[target] = 0
        Policy.__init__(self, target, p, name, alpha)
        
    def first(self, sample):
        self.obs_sample = sample
        self.candidates = set(utils.all_but(self.target, self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result, intervention_sample):
        # Remove identified parents
        self.candidates.difference_update(intersection(result.accepted))
        # Empty-set strategy 2.0
        last_intervention = self.interventions[-1]
        if set() in icp.icp([self.obs_sample, intervention_sample], self.target, self.alpha, selection=[set()]).accepted:
            self.candidates.difference_update({last_intervention})
        # Pick next intervention
        self.current_ratios = ratios(self.p, result.accepted)
        var = self.pick_intervention()
        self.interventions.append(var)
        return var
    
    def pick_intervention(self):
        below_half = set()
        for i,r in enumerate(self.current_ratios):
            if r < 0.5:
                below_half.add(i)
        choice = set.difference(self.candidates, below_half)
        if len(choice) == 0:
            None
        else:
            return np.random.choice(list(choice))
    
class Markov(Policy):
    """Markov policy: selects variables at random from Markov blanket
    estimate.
    """
    def __init__(self, target, p, name, alpha):
        Policy.__init__(self, target, p, name, alpha)
        
    def first(self, sample):
        self.candidates = set(markov_blanket(sample, self.target))
        var = self.pick_intervention()
        return var

    def next(self, result, _):
        # Remove identified parents
        self.candidates.difference_update(intersection(result.accepted))
        # Pick next intervention
        var = self.pick_intervention()
        return var
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            return np.random.choice(list(self.candidates))

class MarkovE(Policy):
    """Markov + empty-set: selects variables at random from Markov blanket
    estimate, removes variables for which, after being intervened, the
    empty set is accepted
    """    
    def __init__(self, target, p, name, alpha):
        Policy.__init__(self, target, p, name, alpha)
        
    def first(self, sample):
        self.obs_sample = sample
        self.candidates = set(markov_blanket(sample, self.target))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result, intervention_sample):
        # Remove identified parents
        self.candidates.difference_update(intersection(result.accepted))
        # Empty-set strategy 2.0
        last_intervention = self.interventions[-1]
        if set() in icp.icp([self.obs_sample, intervention_sample], self.target, self.alpha, selection=[set()]).accepted:
            self.candidates.difference_update({last_intervention})
        # Pick next intervention
        var = self.pick_intervention()
        self.interventions.append(var)
        return var
    
    def pick_intervention(self):
        if len(self.candidates) == 0:
            return None
        else:
            return np.random.choice(list(self.candidates))

class MarkovR(Policy):
    """Markov + ratio: selects variables from Markov blanket
    estimate, that have a stability ratio above 1/2.
    """
    def __init__(self, target, p, name, alpha):
        self.current_ratios = np.ones(p) * 0.5
        self.current_ratios[target] = 0
        Policy.__init__(self, target, p, name, alpha)
        
    def first(self, sample):
        self.candidates = set(markov_blanket(sample, self.target)) # set(range(self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result, _):
        # Remove identified parents
        self.candidates.difference_update(intersection(result.accepted))
        # Pick next intervention
        self.current_ratios = ratios(self.p, result.accepted)
        var = self.pick_intervention()
        self.interventions.append(var)
        return var
    
    def pick_intervention(self):
        below_half = set()
        for i,r in enumerate(self.current_ratios):
            if r < 0.5:
                below_half.add(i)
        choice = set.difference(self.candidates, below_half)
        if len(choice) == 0:
            None
        else:
            return np.random.choice(list(choice))

class MarkovER(Policy):
    """Markov + empty-set + ratio: selects variables at random from Markov
    blanket estimate which have a stability ratio above 1/2. Removes
    variables for which, after being intervened, the empty set is
    accepted.
    """
    def __init__(self, target, p, name, alpha):
        self.current_ratios = np.ones(p) * 0.5
        self.current_ratios[target] = 0
        Policy.__init__(self, target, p, name, alpha)
        
    def first(self, sample):
        self.obs_sample = sample
        self.candidates = set(markov_blanket(sample, self.target)) # set(range(self.p))
        var = self.pick_intervention()
        self.interventions.append(var)
        return var

    def next(self, result, intervention_sample):
        # Remove identified parents
        self.candidates.difference_update(intersection(result.accepted))
        # Empty-set strategy 2.0
        last_intervention = self.interventions[-1]
        if set() in icp.icp([self.obs_sample, intervention_sample], self.target, self.alpha, selection=[set()]).accepted:
            self.candidates.difference_update({last_intervention})
        # Pick next intervention
        self.current_ratios = ratios(self.p, result.accepted)
        var = self.pick_intervention()
        self.interventions.append(var)
        return var
    
    def pick_intervention(self):
        below_half = set()
        for i,r in enumerate(self.current_ratios):
            if r < 0.5:
                below_half.add(i)
        choice = set.difference(self.candidates, below_half)
        if len(choice) == 0:
            None
        else:
            return np.random.choice(list(choice))

