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
from src import sampling, policy, utils
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
        W, ordering = sampling.dag_avg_deg(p, k, w_min, w_max)
        target = np.random.choice(range(p))
        parents,_,_,mb = utils.graph_info(target, W)
        if len(parents) > 0 and len(parents) != len(mb):
            sem = sampling.LGSEM(W, ordering, (var_min, var_max), (int_min, int_max))
            (truth, _, _, _) = utils.graph_info(target, W)
            cases.append(policy.TestCase(i, sem, target, truth))
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

def wrapper(parameters):
    """ Wrapper function for running "run_policy" on the pool of workers"""
    result = policy.run_policy(**parameters)
    return result

def evaluate_policies(cases, runs, policies, names, batch_size=round(1e4), n=round(1e5), alpha=0.01, population=False, max_iter=100, random_state=None, debug=False, n_workers=None):
    """Evaluate the given policies over the given cases (SCMs) over runs with different random seeds, using as many cores as possible"""
    # # Multiprocessing support
    # if not __name__ == '__main__':
    #     raise Exception("Not in __main__module. Name = ", __name__)
    # Prepare experiments: Each "experiment" is a single run of a policy over an SCM
    start = time.time()
    print("Compiling experiment batch...", end="")
    experiments = []
    for i, policy in enumerate(policies):
        for run in range(runs):
            for case in cases:
                parameters = {'case':case,
                              'policy': policy,
                              'name': names[i],
                              'n': n,
                              'alpha': alpha,
                              'population': population,
                              'max_iter': max_iter,
                              'debug': debug,
                              'random_state': np.random.randint(999999)}
                experiments.append(parameters)
    print("  done (%0.2f seconds)" % (time.time() - start))
    n_exp = len(experiments)
    # Run experiments in batches to prevent memory explosion due to
    # large interables with pool.map
    if n_workers is None:
        n_workers = os.cpu_count()
    print("Available cores: %d" % os.cpu_count())
    print("Running a total of %d experiments with %d workers in batches of size %d" % (n_exp, n_workers, batch_size))
    setting = "Population" if population else "Finite (%d samples/environment)" % n
    print("%s setting with a maximum of %d iterations per experiment" % (setting, max_iter))
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
            result += pool.map(wrapper, batch, chunksize=1)
        else:
            result += map(wrapper, batch)
        batch_end = time.time()
        print("  %d/%d experiments completed (%0.2f seconds)" % ((i+1)*batch_size, n_exp, batch_end-batch_start))
    # Process the results into a more friendly format which can then
    # be used by the notebooks for plotting
    return process_results(result, len(policies), runs, len(cases))
