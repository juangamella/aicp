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

import time
import numpy as np
from functools import reduce
from src import icp, population_icp, utils, normal_distribution

# --------------------------------------------------------------------
# Policy evaluation

def evaluate_policy(policy, cases, name=None, n=round(1e5), population=False,  random_seed=42, debug=False):
    """Evaluate a policy over the given test cases, returning a
    PolicyEvaluationResults object containing the results
    """
    np.random.seed(random_seed)
    results = []
    for i,case in enumerate(cases):
        print("%0.2f%% Evaluating policy \"%s\" on test case %d..." % (i/len(cases)*100, name, i)) if debug else None
        result = run_policy(policy, case, name=name, n=n, population=population, debug=debug)
        print(" done (%0.2f seconds). truth: %s estimate: %s" % (result.time, case.truth, result.estimate)) if debug else None
        results.append(result)
    return results

def run_policy(policy, case, name=None, n=round(1e5), population=False, debug=False):
    """Execute a policy over a given test case, recording the results"""
    # Initialization
    policy = policy(case.target, case.sem.p, name=name)
    icp = population_icp.population_icp if population else icp.icp
    history = []
    # Generate observational samples
    e = case.sem.sample(n, population)
    envs = [e]
    start = time.time()
    (next_intervention, current_estimate) = policy.first(e)
    history.append((None, current_estimate, next_intervention))
    i = 1
    while next_intervention is not None:
        print("  %d current estimate: %s next intervention: %s" % (i, current_estimate, next_intervention)) if debug else None
        new_env = case.sem.sample(n, population, noise_interventions = next_intervention)
        envs.append(new_env)
        result = icp(envs, case.target, debug=False)
        (next_intervention, current_estimate) = policy.next(result)
        history.append((result, current_estimate, next_intervention))
        i += 1
    end = time.time()
    elapsed = end - start
    # Return result
    return EvaluationResult(policy, case, current_estimate, history, elapsed)
    
def jaccard_distance(A, B):
    """Compute the jaccard distance between sets A and B"""
    return len(set.intersection(A,B)) / len(set.union(A,B))


class EvaluationResult():
    """Class to contain all information resulting from evaluating a policy
    over a test case"""

    def __init__(self, policy, case, estimate, history, time):
        self.policy = policy
        self.case = case
        # Info
        self.estimate = estimate # estimate produced by the policy
        self.history = history # interventions and intermediate results of the policy
        self.time = time # time used by the policy

    def estimates(self):
        """Return the parents estimated by the policy at each step"""
        return list(map(lambda step: step[1], self.history))

    def interventions(self):
        """Return the interventions carried out by the policy"""
        return list(map(lambda step: step[2], self.history[0:len(self.history)-1]))

    def intervened_variables(self):
        """Return the intervened variables"""
        return list(map(lambda step: step[2][0,0], self.history[0:len(self.history)-1]))
    
class TestCase():
    """Object that represents a test case
    ie. SEM + target + expected result
    """
    def __init__(self, sem, target, truth):
        self.sem = sem
        self.target = target
        self.truth = truth

# --------------------------------------------------------------------
# Policies

class Policy():
    def __init__(self, target, p, name=None):
        self.target = target
        self.p = p # no. of variables
        self.name = name

    def first(self, observational_data):
        """Returns the initial intervention and initial estimate"""
        return (None, set())

    def next(self, icp_results):
        """Returns the next intervention and the current estimate"""
        return (None, set())

class RandomPolicy(Policy):
    """Random policy: selects a previously unintervened variable at
    random, until the only accepted set is the estimate or until all
    variables have been intervened on.
    """
    def __init__(self, target, p, name):
        self.interventions = []
        Policy.__init__(self, target, p, name)
    
    def first(self, _):
        return (self.random_intervention(), set())

    def next(self, result):
        if [result.estimate] == result.accepted:
            return (None, result.estimate)
        else:
            return (self.random_intervention(), result.estimate)

    def random_intervention(self):
        choice = [i for i in range(self.p) if i not in self.interventions and i != self.target]
        if not choice: # ie. all variables already intervened on
            return None
        else:
            i = np.random.choice(choice)
            self.interventions.append(i)
            return np.array([[i, 0, 10]])

class MBPolicy(Policy):
    """Markov Blanket policy: only considers subsets (and intervenes on
    variables) of the Markov blanket """

    def __init__(self, target, p, name, alpha=0.01):
        self.interventions = []
        self.alpha = alpha # significance level for estimating the MB
        Policy.__init__(self, target, p, name)

    def first(self, e):
        if isinstance(e, normal_distribution.NormalDistribution):
            self.mb = set(population_icp.markov_blanket(self.target, e))
            return (self.pick_intervention(), set())
        else:
            raise Exception("Not yet implemented")

    def next(self, result):
        # Estimate is the intersection of all sets which are subsets
        # of the Markov blanket
        accepted = list(filter(lambda s: set.intersection(s, self.mb) == s, result.accepted))
        estimate = reduce(lambda acc, s: set.intersection(acc, s), accepted, set(range(self.p)))
        if [estimate] == accepted:
            return (None, estimate)
        else:
            return (self.pick_intervention(), estimate)
    
    def pick_intervention(self):
        choice = [i for i in self.mb if i not in self.interventions]
        if not choice: # ie. all variables already intervened on
            return None
        else:
            i = np.random.choice(choice)
            self.interventions.append(i)
            return np.array([[i, 0, 10]])

class SBPopPolicy(Policy):
    """Using stable blanket and derived theoretical results"""

    def __init__(self, target, p, name, alpha=0.01, full=False):
        self.interventions = []
        self.full = full
        Policy.__init__(self, target, p, name)

    def first(self, e):
        self.mb = set(population_icp.markov_blanket(self.target, e))
        self.sb = self.mb.copy()
        return (self.pick_intervention(), set())

    def next(self, result):
        if self.full:
            raise Exception("Not yet implemented")
        else:
            # Estimate is the intersection of all sets which are subsets
            # of the stable blanket
            self.sb = population_icp.stable_blanket(result.accepted, result.mses)
            accepted = list(filter(lambda s: set.intersection(s, self.sb) == s, result.accepted))
            estimate = reduce(lambda acc, s: set.intersection(acc, s), accepted, set(range(self.p)))
            if [estimate] == accepted:
                return (None, estimate)
            else:
                return (self.pick_intervention(), estimate)

    def pick_intervention(self):
        choice = [i for i in self.sb if i not in self.interventions]
        if not choice: # ie. all variables already intervened on
            return None
        else:
            i = np.random.choice(choice)
            self.interventions.append(i)
            return np.array([[i, 0, 10]])
