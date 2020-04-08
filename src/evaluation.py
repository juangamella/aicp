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

import time
import numpy as np
from src import sampling, utils, icp, population_icp
import multiprocessing
import os

def gen_cases(n, P, k, w_min=1, w_max=1, var_min=1, var_max=1, int_min=0, int_max=0, random_state=39):
    """
    Generate random experimental cases (ie. linear SEMs). Parameters:
      - n: total number of cases
      - P: number of variables in the SEMs (either an integer or a tuple to indicate a range)
      - w_min, w_max: Weights of the SEMs are sampled at uniform between w_min and w_max
      - var_min, var_max: Weights of the SEMs are sampled at uniform between var_min and var_max
      - int_min, int_max: Weights of the SEMs are sampled at uniform between int_min and int_max
      - random_state: to fix the random seed for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)
    cases = []
    i = 0
    while i < n:
        if isinstance(P, tuple):
            p = np.random.randint(P[0], P[1]+1)
        else:
            p = P
        W = sampling.dag_avg_deg(p, k, w_min, w_max)
        target = np.random.choice(range(p))
        parents,_,_,mb = utils.graph_info(target, W)
        if len(parents) > 0 and len(parents) != len(mb):
            sem = sampling.LGSEM(W, (var_min, var_max), (int_min, int_max))
            (truth, _, _, _) = utils.graph_info(target, W)
            cases.append(TestCase(i, sem, target, truth))
            i += 1
    return cases

def process_results(unprocessed, P, R, G):
    """Process the results returned by the worker pool, sorting them by
    policy and run e.g. results[i][j][k] are the results from policy i
    on run j on graph k. Parameters:
      - unprocessed: Unprocessed results (as returned by the worker pool)
      - P: number of policies
      - R: number of runs
      - G: number of graphs/SCMs/test cases
    """
    results = []
    for i in range(P):
        policy_results = []
        for r in range(R):
            run_results = unprocessed[(i*G*R + G*r):(i*G*R + G*(r+1))]
            policy_results.append(run_results)
        results.append(policy_results)
    return results

def evaluate_policies(cases, policies, policy_names, runs, params, n_workers=None, batch_size=round(1e4), debug=False):
    """Evaluate the given policies over the given cases (SCMs) over runs
    with different random seeds, using as many cores as possible"""

    start = time.time()
    print("Compiling experiment batch...", end="")
    experiments = []
    for i, policy in enumerate(policies):
        for run in range(runs):
            for case in cases:
                experiment = ExperimentSettings(
                    case = case,
                    policy = policy,
                    policy_name = policy_names[i],
                    random_state = np.random.randint(999999),
                    **params)
                experiments.append(experiment)
    print("  done (%0.2f seconds)" % (time.time() - start))
    n_exp = len(experiments)
    # Run experiments in batches to prevent memory explosion due to
    # large interables with pool.map
    if n_workers is None:
        n_workers = os.cpu_count() - 1
    print("Available cores: %d" % os.cpu_count())
    print("Running a total of %d experiments with %d workers in batches of size %d" % (n_exp, n_workers, batch_size))
    setting = "Population" if params['population'] else ("Finite (%d samples/environment and %s obs. samples)" % (params['n_int'], params['n_obs']))
    print("%s setting with a maximum of %d iterations per experiment" % (setting, params['max_iter']))
    pool = multiprocessing.Pool(n_workers)
    n_batches = int(np.floor(n_exp / batch_size) + (n_exp % batch_size != 0))
    result = []
    for i in range(n_batches):
        if i == n_batches-1:
            batch = experiments[i*batch_size::]
        else:
            batch = experiments[i*batch_size:(i+1)*batch_size]
        batch_start = time.time()
        if n_workers > 1:
            result += pool.map(run_policy, batch, chunksize=1)
        else:
            result += list(map(run_policy, batch))
        batch_end = time.time()
        print("  %d/%d experiments completed (%0.2f seconds)" % ((i+1)*batch_size, n_exp, batch_end-batch_start))
    # Process the results into a more friendly format which can then
    # be used by the notebooks for plotting
    return process_results(result, len(policies), runs, len(cases))

def run_policy(settings):
    """Execute a policy over a given test case, returning a returning a
    PolicyEvaluationResults object containing the result"""

    print(vars(settings)) if settings.debug else None
    
    # Initialization
    case = settings.case
    if settings.random_state is not None:
        np.random.seed(settings.random_state)
    policy = settings.policy(case.target, case.sem.p, name=settings.policy_name)
    history = [] # where we store the ICP estimate / next intervention

    # Generate observational samples
    if settings.population:
        e = case.sem.sample(population = True)
        envs = [e]
    else:
        e = case.sem.sample(n = settings.n_obs)
        envs = Environments(case.sem.p, e)
    start = time.time()

    # Initial iteration
    next_intervention = policy.first(e)
    current_estimate = set()
    result = None
    selection = 'all' # on the first iteration, evaluate all possible candidate sets
    i = 1

    # Remaining iterations
    while current_estimate != case.truth and i <= settings.max_iter:
        history.append((current_estimate, next_intervention, len(selection)))
        print(" (case_id: %s, target: %d, truth: %s, policy: %s) %d current estimate: %s accepted sets: %d next intervention: %s" % (case.id, case.target, case.truth, policy.name, i, current_estimate, len(selection), next_intervention)) if settings.debug else None
        if next_intervention is not None:
            # Perform intervention
            noise_intervention = np.array([[next_intervention, settings.intervention_mean, settings.intervention_var]])
            if settings.population:
                new_env = case.sem.sample(population = True, noise_interventions = noise_intervention)
                envs.append(new_env)
                result = population_icp.population_icp(envs, case.target, selection=selection, debug=False)
            else:
                new_env = case.sem.sample(n = settings.n_int, noise_interventions = noise_intervention)
                envs.add(next_intervention, new_env)
                result = icp.icp(envs.to_list(), case.target, selection=selection, alpha=settings.alpha, debug=False)
            current_estimate = result.estimate
            selection = result.accepted # in each iteration we only need to run ICP on the sets accepted in the previous one
            next_intervention, selection = policy.next(result)
        i += 1

    # Return result
    end = time.time()
    if i > settings.max_iter:
        print(" (case_id: %s, target: %d, truth: %s, policy: %s) reached %d > %d iterations" % (case.id, case.target, case.truth, policy.name, i, settings.max_iter)) if settings.debug else None
    elapsed = end - start
    print("  (case_id: %s) done (%0.2f seconds)" % (case.id, elapsed)) if settings.debug else None
    return EvaluationResult(policy.name, current_estimate, history)

# --------------------------------------------------------------------
# Class definitions

class TestCase():
    """Object that represents a test case
    ie. SCM + target + expected result
    """
    def __init__(self, id, sem, target, truth):
        self.id = id
        self.sem = sem
        self.target = target
        self.truth = truth

class ExperimentSettings():
    """Contains the parameters for a single experiment
    ie. one run of one policy over one test case (SCM)"""
    
    def __init__(self,
                 case,
                 policy,
                 policy_name,
                 population,
                 debug,
                 max_iter,
                 intervention_mean,
                 intervention_var,
                 random_state,
                 n_int = None,
                 n_obs = None,
                 alpha = None):
                
        # Build object
        self.case = case
        self.policy = policy
        self.policy_name = policy_name
        self.population = population
        self.debug = debug
        self.max_iter = max_iter
        self.intervention_mean = intervention_mean
        self.intervention_var = intervention_var
        self.random_state = random_state
        
        if not population:
            if n_int is not None and n_obs is not None and alpha is not None:
                self.n_int = n_int
                self.n_obs = n_obs
                self.alpha = alpha
            else:
                raise Exception("Missing parameters for finite sample setting")

class EvaluationResult():
    """Class to contain all information resulting from evaluating a policy
    over a test case"""

    def __init__(self, policy, estimate, history):
        self.policy = policy
        #self.case = case
        # Info
        self.estimate = estimate # estimate produced by the policy
        self.history = history # interventions and intermediate results of the policy
        #self.time = time # time used by the policy

    def estimates(self):
        """Return the parents estimated by the policy at each step"""
        return list(map(lambda step: step[0].estimate, self.history))

    def interventions(self):
        """Return the interventions carried out by the policy"""
        return list(map(lambda step: step[1], self.history))

    def intervened_variables(self):
        """Return the intervened variables"""
        return list(map(lambda step: step[1][0,0], self.history))
            
class Environments():
    """Class to hold samples from different environments, merging if they
    arise from the same intervention to improve efficiency"""
    def __init__(self, p, e):
        self.envs = dict([(i,[]) for i in range(p)])
        self.envs[None] = e

    def add(self, target, env):
        # Merge samples that arise from same interventional
        # environment
        if self.envs[target] == []:
            self.envs[target] = env
        else:
            self.envs[target] = np.vstack([env, self.envs[target]])

    def to_list(self):
        envs = []
        for env in self.envs.values():
            if env != []:
                envs.append(env)
        return envs
