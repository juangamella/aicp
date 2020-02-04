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
import sys
import argparse
import os

def save_results(results, filename=None):
    if filename is None:
        filename = "experiments/results_%d.pickle" % time.time()
    f = open(filename, "wb")
    pickle.dump(results, f)
    f.close()
    return filename

def gen_cases(n, P, k, w_min=1, w_max=1, var_min=1, var_max=1, int_min=0, int_max=0, random_state=39):
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

# --------------------------------------------------------------------
# Multiprocessing support

def wrapper(parameters):
    result = policy.run_policy(**parameters)
    return result

def evaluate_policies(cases, runs, policies, names, batch_size=round(1e4), n=round(1e5), alpha=0.01, population=False, max_iter=100, random_state=None, debug=False, n_workers=None):
    """Evaluate a policy over the given test cases, using as many cores as possible"""
    if not __name__ == '__main__':
        raise Exception("Not in __main__module. Name = ", __name__)
    # Prepare experiments
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
    return result
    
# Default settings
arguments = {
    'n_workers': {'default': 1, 'type': int},
    'batch_size': {'default': 50, 'type': int},
    'debug': {'default': False, 'type': bool}, # False
    'avg_deg': {'default': 3, 'type': float},
    'G': {'default': 4, 'type': int},
    'runs': {'default': 1, 'type': int},
    'n_min': {'default': 8, 'type': int},
    'n_max': {'default': 8, 'type': int},
    'w_min': {'default': 0.1, 'type': float},
    'w_max': {'default': 0.2, 'type': float},
    'var_min': {'default': 0.1, 'type': float},
    'var_max': {'default': 1, 'type': float},
    'int_min': {'default': 0, 'type': float},
    'int_max': {'default': 1, 'type': float},
    'random_state': {'default': 42, 'type': int},
    'finite': {'default': False, 'type': bool}, # False
    'max_iter': {'default': -1, 'type': int}, # -1
    'n': {'default': 100, 'type': int},
    'alpha': {'default': 0.01, 'type': float}}

# Settings from input
parser = argparse.ArgumentParser(description='Run experiments')
for name, params in arguments.items():
    if params['type']==bool:
        action = {'action': 'store_true'}
    else:
        action = {'action': 'store', 'type': params['type']}
    parser.add_argument("--" + name,
                        dest=name,
                        required=False,
                        default=params['default'],
                        **action)

args = parser.parse_args()
print()
print(args)
print()

# --------------------------------------------------------------------
# Compile experiments

if args.max_iter == -1:
    max_iter = args.n_max
else:
    max_iter = args.max_iter
    
P = args.n_min if args.n_min == args.n_max else (args.n_min, args.n_max)

n_workers = None if args.n_workers == -1 else args.n_workers

# Generate test cases
cases = gen_cases(args.G,
                  P,
                  args.avg_deg,
                  args.w_min,
                  args.w_max,
                  args.var_min,
                  args.var_max,
                  args.int_min,
                  args.int_max,
                  args.random_state)

# --------------------------------------------------------------------
# Evaluate experiments

start = time.time()
print("\n\nBeggining experiments on %d graphs at %s\n\n" % (len(cases), datetime.now()))

population = not args.finite

if population:
    policies = [policy.MBPolicy, policy.RatioPolicy, policy.RandomPolicy]
    names = ["markov blanket", "ratio policy", "random"]
else:
    policies = [policy.RandomPolicyF, policy.MarkovPolicyF, policy.ProposedPolicyMEF, policy.ProposedPolicyMERF, policy.ProposedPolicyEF, policy.ProposedPolicyERF]
    names = ["random", "markov", "markov + e", "markov + e + r", "e", "e + r"]
    
evaluation_params = {'population': population,
                     'n': args.n,
                     'alpha': args.alpha,
                     'debug': args.debug,
                     'random_state': None,
                     'max_iter': max_iter,
                     'n_workers': n_workers,
                     'batch_size': args.batch_size}

results = evaluate_policies(cases, args.runs, policies, names, **evaluation_params)
end = time.time()
print("\n\nFinished experiments at %s (elapsed %0.2f seconds)" % (datetime.now(), end-start))

# Save results

processed_results = []
R = args.runs
G = len(cases)
P = len(policies)
for i in range(P):
    policy_results = []
    for r in range(R):
        run_results = results[(i*G*R + G*r):(i*G*R + G*(r+1))]
        policy_results.append(run_results)
    processed_results.append(policy_results)

filename = "experiments/results_%d" % end
for k,v in vars(args).items():
    filename = filename + "_" + k + ":" + str(v)
filename = filename + ".pickle"

save_results([cases] + processed_results, filename)
print("Saved to file \"%s\"" % filename)
