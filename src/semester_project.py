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

import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src import sampling, policy, utils
import multiprocessing

def save_results(results, filename=None):
    if filename is None:
        filename = "experiments/results_%d.pickle" % time.time()
    f = open(filename, "wb")
    pickle.dump(results, f)
    pickle.dump(results, f)
    f.close()
    return filename

def load_results(filename):
    f = open(filename, "rb")
    return pickle.load(f)

def gen_cases(n, p, k, w_min=1, w_max=1, var_min=1, var_max=1, int_min=0, int_max=0, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    cases = []
    for i in range(n):
        W, ordering = sampling.dag_avg_deg(p, k, w_min, w_max)
        sem = sampling.LGSEM(W, ordering, (var_min, var_max), (int_min, int_max))
        target = np.random.choice(range(p))
        (truth, _, _, _) = utils.graph_info(target, W)
        cases.append(policy.TestCase(i, sem, target, truth))
    return cases

# --------------------------------------------------------------------
# Multiprocessing support

def wrapper(parameters):
    result = policy.run_policy(**parameters)
    print("  case %d done" % parameters['case'].id)
    return result

def evaluate_policy(policy, cases, name=None, n=round(1e5), population=False, max_iter=100, random_state=None, debug=False, n_workers=4):
    """Evaluate a policy over the given test cases, using as many cores as possible"""
    start = time.time()
    print("Evaluating policy \"%s\" with %d workers... " % (name, n_workers))
    iterable = []
    for case in cases:
        parameters = {'case':case,
                      'policy': policy,
                      'name': name,
                      'n': n,
                      'population': population,
                      'max_iter': max_iter,
                      'debug': debug}
        iterable.append(parameters)
    if __name__ == '__main__':
        p = multiprocessing.Pool(n_workers)
        result = p.map(wrapper, iterable)
        end = time.time()
        print("  done (%0.2f seconds)" % (end-start))
        return result
    else:
        raise Exception("Not in __main__module. Name = ", __name__)
    
use_results = None
use_parallelism = False
debug = True

# --------------------------------------------------------------------
# Run or load experiments

if use_results is None:
    N = 64
    runs = 8
    
    # Generate test cases
    cases = gen_cases(N, 12, 2, 0.1, 1)

    # Evaluate
    
    start = time.time()
    print("\n\nBeggining experiments at %s\n\n" % datetime.now())

    pop_rand_results = []
    pop_mb_results = []
    pop_ratio_results = []    

    if use_parallelism and __name__ == '__main__':
        print("Running in parallel")
        evaluation_func = evaluate_policy
    elif use_parallelism:
        print("Not in __main__ module, running sequentially")
        evaluation_func = policy.evaluate_policy
    else:
        print("Running sequentially")
        evaluation_func = policy.evaluate_policy
        
    for i in range(runs):
        print("--- RUN %d ---" % i)
        pop_mb_results.append(evaluation_func(policy.MBPolicy, cases, name="markov blanket", population=True, debug=debug, random_state=None))
        pop_ratio_results.append(evaluation_func(policy.RatioPolicy, cases, name="ratio policy", population=True, debug=debug, random_state=None))
        pop_rand_results.append(evaluation_func(policy.RandomPolicy, cases, name="random", population=True, debug=debug, random_state=None))

    end = time.time()
    print("\n\nFinished experiments at %s (elapsed %0.2f seconds)" % (datetime.now(), end-start))
    
    # Save results
    results = [pop_mb_results, pop_ratio_results, pop_rand_results]
    filename = save_results(results)
    print("Saved to file \"%s\"" % filename)

else:
    # --------------------------------------------------------------------
    # Plotting
    
    colors = ["#ff7878", "#ffbc78", "#ffff78", "#bcff78", "#78ffbc"]
    markers = ["o", "*", "+"]
    
    runs = len(results[0])
    N = len(results[0][0])
    
    no_ints = np.zeros((len(results), runs, N))

    for k, policy_runs in enumerate(results):
        name = policy_runs[0][0].policy.name
        print("Plotting intervention numbers for policy %d: %s" % (k, name))
        for i,run_results in enumerate(policy_runs):
            no_ints[k, i,:] = list(map(lambda result: len(result.interventions()), run_results))
            plt.scatter(np.arange(N), no_ints[k, i,:], c=colors[k], marker=markers[k])
    plt.legend()
    plt.show(block=False)
    print(no_ints.mean(axis=1))
