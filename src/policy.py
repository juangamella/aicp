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
from src import icp, population_icp

# --------------------------------------------------------------------
# Policy evaluation

def evaluate_policy(policy, cases, name=None, n=round(1e5), population=False,  random_seed=42, debug=False):
    """Evaluate a policy over the given test cases, returning a
    PolicyEvaluationResults object containing the results
    """
    np.random.seed(random_seed)
    results = []
    for i,case in enumerate(cases):
        print("%0.2f%% Evaluating policy on test case %d..." % (i/len(cases)*100, i), end="") if debug else None
        result = run_policy(policy, case, name=name, n=n, population=population, debug=debug)
        print(" done (%0.2f seconds)" % result.time) if debug else None
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
    next_intervention = policy.first(e)
    i = 1
    while next_intervention is not None:
        print("  %d: intervention %s" % (i, next_intervention)) if debug else None
        new_env = case.sem.sample(n, population, noise_interventions = next_intervention)
        envs.append(new_env)
        result = icp(envs, case.target, debug=False)
        history.append((next_intervention, result))
        next_intervention = policy.next(result)
        i += 1
    end = time.time()
    elapsed = end - start
    # Return result
    return EvaluationResult(policy, case, result.estimate, history, elapsed)
    
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
        return None

    def next(self, icp_results):
        return None

class RandomPolicy(Policy):
    """Random policy: selects a previously unintervened variable at
    random, until the only accepted set is the estimate or until all
    variables have been intervened on.
    """
    def __init__(self, target, p, name):
        self.interventions = []
        Policy.__init__(self, target, p, name)
    
    def first(self, _):
        return self.random_intervention()

    def next(self, result):
        if [result.estimate] == result.accepted:
            return None
        else:
            return self.random_intervention()

    def random_intervention(self):
        choice = [i for i in range(self.p) if i not in self.interventions and i != self.target]
        if not choice: # ie. all variables already intervened on
            return None
        else:
            i = np.random.choice(choice)
            self.interventions.append(i)
            return np.array([[i, 0, 10]])

