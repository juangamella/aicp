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
from termcolor import colored
from src import icp, population_icp, utils, normal_distribution
import networkx as nx

# --------------------------------------------------------------------
# Policy evaluation        

def evaluate_policy(policy, cases, name=None, n=round(1e5), population=False, max_iter=100, random_state=42, debug=False):
    """Evaluate a policy over the given test cases, in a sequential manner"""
    results = []
    start = time.time()
    print("Evaluating policy \"%s\" sequentially..." % name)
    for i,case in enumerate(cases):
        print("%0.2f%% Evaluating policy \"%s\" on test case %d (truth = %s)..." % (i/len(cases)*100, name, case.id, case.truth)) if debug else None
        result = run_policy(policy, case, name=name, n=n, population=population, max_iter=max_iter, debug=debug, random_state=random_state)
        if debug:
            msg = " done (%0.2f seconds). truth: %s estimate: %s" % (result.time, case.truth, result.estimate)
            color = "green" if case.truth == result.estimate else "red"
            print(colored(msg, color))
        results.append(result)
    end = time.time()
    print("  done (%0.2f seconds)" % (end-start))
    return results

def run_policy(policy, case, name=None, n=round(1e5), max_iter=100, population=True, debug=False, random_state=42):
    """Execute a policy over a given test case, returning a returning a
    PolicyEvaluationResults object containing the result"""
    # Initialization
    if random_state is not None:
        np.random.seed(random_state)
    policy = policy(case.target, case.sem.p, name=name)
    icp = population_icp.population_icp if population else icp.icp
    history = [] # store the ICP result and next intervention
    # Generate observational samples
    e = case.sem.sample(n, population)
    envs = [e]
    start = time.time()
    # Initial iteration
    next_intervention = policy.first(e)
    current_estimate = None
    result = None
    selection = 'all' # on the first iteration, evaluate all possible candidate sets
    i = 1
    ####
    parents, _, _, _ = utils.graph_info(case.target, case.sem.W)
    ####
    while current_estimate != case.truth and i <= max_iter:
        history.append((current_estimate, next_intervention))
        print(" (case_id: %s, target: %d, truth: %s, policy: %s) %d current estimate: %s accepted sets: %d next intervention: %s" % (case.id, case.target, case.truth, policy.name, i, current_estimate, len(selection), next_intervention)) if debug else None
        # Perform intervention
        new_env = case.sem.sample(n, population, noise_interventions = next_intervention)
        envs.append(new_env)
        # Run ICP
        result = icp(envs, case.target, selection=selection, debug=False)
        current_estimate = result.estimate
        selection = result.accepted # in each iteration we only need to run ICP on the sets accepted in the previous one
        next_intervention = policy.next(result)
        ######
        var_ratios = ratios(case.sem.p, result.accepted)
        for j,r in enumerate(var_ratios):
            if j in parents and r < 0.5:
                print(parents)
                print(var_ratios)
                print("Hypothesis is false! (case_id: %s, target: %d, interventions: %s)" % (case.id, case.target, history))
        ######
        i += 1
    end = time.time()
    if i > max_iter:
        print(" (case_id: %s, target: %d, truth: %s, policy: %s) reached %d > %d iterations" % (case.id, case.target, case.truth, policy.name, i, max_iter))
    elapsed = end - start
    print("  (case_id: %s) done (%0.2f seconds)" % (case.id, elapsed)) if debug else None
    # Return result
    return EvaluationResult(policy, case, current_estimate, history, elapsed)
    
def jaccard_distance(A, B):
    """Compute the jaccard distance between sets A and B"""
    if len(A) == 0 and len(B) == 0:
        return 1
    else:
        return len(set.intersection(A,B)) / len(set.union(A,B))

def ratios(p, accepted):    
    one_hot = np.zeros((len(accepted), p))
    for i,s in enumerate(accepted):
        one_hot[i, list(s)] = 1
    return one_hot.sum(axis=0) / len(accepted)

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
        return list(map(lambda step: step[0].estimate, self.history))

    def interventions(self):
        """Return the interventions carried out by the policy"""
        return list(map(lambda step: step[1], self.history))

    def intervened_variables(self):
        """Return the intervened variables"""
        return list(map(lambda step: step[1][0,0], self.history))
    
class TestCase():
    """Object that represents a test case
    ie. SEM + target + expected result
    """
    def __init__(self, id, sem, target, truth):
        self.id = id
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
        """Returns the initial intervention"""
        return None

    def next(self, icp_results):
        """Returns the next intervention"""
        return None

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
        return self.random_intervention()

    def random_intervention(self):
        var = self.idx[self.i]
        self.i = (self.i+1) % len(self.idx)
        return np.array([[var, 0, 10]])

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
        return self.pick_intervention()
    
    def pick_intervention(self):
        var = self.mb[self.i]
        self.i = (self.i+1) % len(self.mb)
        return np.array([[var, 0, 10]])

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
        return np.array([[var, 0, 10]])

    def next(self, result):
        #print("ratios = %s candidates = %s, interventions = %s" % (self.current_ratios, self.candidates, self.interventions))
        new_ratios = ratios(self.p, result.accepted)
        diff = new_ratios - self.current_ratios
        last_intervention = self.interventions[-1]
        #print(new_ratios)
        for i,r in enumerate(new_ratios):
            if r < 0.5 and i in self.candidates:
                #print("removing %d" % i)
                self.candidates.remove(i)
        self.current_ratios = new_ratios
        return self.pick_intervention()

    def pick_intervention(self):
        choice = set.difference(self.candidates, set(self.interventions))
        if choice == set(): # Have intervened on all variables or there are no parents
            None
        else:
            var = np.random.choice(list(choice))
            self.interventions.append(var)
            return np.array([[var, 0, 10]])
    


