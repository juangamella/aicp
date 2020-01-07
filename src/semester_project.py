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

def gen_cases(n, P, k, w_min=1, w_max=1, var_min=1, var_max=1, int_min=0, int_max=0, random_state=39):
    if random_state is not None:
        np.random.seed(random_state)
    cases = []
    i = 0
    while i < n:
        if isinstance(P, tuple):
            p = np.random.randint(P[0], P[1])
        else:
            p = P
        print("Generating case %d with %d nodes" % (i,p))
        W, ordering = sampling.dag_avg_deg(p, k, w_min, w_max)
        target = np.random.choice(range(p))
        parents = utils.graph_info(target, W)[0]
        if len(parents) > 0:
            sem = sampling.LGSEM(W, ordering, (var_min, var_max), (int_min, int_max))
            (truth, _, _, _) = utils.graph_info(target, W)
            cases.append(policy.TestCase(i, sem, target, truth))
            i += 1
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
                      'debug': debug,
                      'random_state': np.random.randint(999999)}
        iterable.append(parameters)
    if __name__ == '__main__':
        p = multiprocessing.Pool(n_workers)
        result = p.map(wrapper, iterable)
        end = time.time()
        print("  done (%0.2f seconds)" % (end-start))
        return result
    else:
        raise Exception("Not in __main__module. Name = ", __name__)


# Default settings
arguments = {
    'n_workers': {'default': 4, 'type': int},
    'use_parallelism': {'default': True, 'type': bool},
    'debug': {'default': True, 'type': bool},
    'avg_deg': {'default': 2, 'type': float},
    'G': {'default': 10, 'type': int},
    'runs': {'default': 1, 'type': int},
    'n_min': {'default': 8, 'type': int},
    'n_max': {'default': 8, 'type': int},
    'w_min': {'default': 1, 'type': int},
    'w_max': {'default': 2, 'type': int},
    'var_min': {'default': 0.1, 'type': int},
    'var_max': {'default': 1, 'type': int},
    'int_min': {'default': 0, 'type': int},
    'int_max': {'default': 1, 'type': int},
    'random_state': {'default': 42, 'type': int}}

# Settings from input
parser = argparse.ArgumentParser(description='Run experiments')
for name, args in arguments.items():
    parser.add_argument("--" + name,
                        action="store",
                        dest=name,
                        nargs=1,
                        type=args['type'],
                        required=False,
                        default=args['default'])

args = parser.parse_args()
print()
print(args)
print()

# --------------------------------------------------------------------
# Run experiments
max_iter = args.n_max
debug = args.debug

P = args.n_min if args.n_min == args.n_max else (args.n_min, args.n_max)

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

# Evaluate

start = time.time()
print("\n\nBeggining experiments at %s\n\n" % datetime.now())

pop_rand_results = []
pop_mb_results = []
pop_ratio_results = []    

if args.use_parallelism and __name__ == '__main__':
    print("Running in parallel")
    evaluation_func = evaluate_policy
elif args.use_parallelism:
    print("Not in __main__ module, running sequentially")
    evaluation_func = policy.evaluate_policy
else:
    print("Running sequentially")
    evaluation_func = policy.evaluate_policy

for i in range(args.runs):
    print("--- RUN %d ---" % i)
    pop_mb_results.append(evaluation_func(policy.MBPolicy, cases, name="markov blanket", population=True, debug=debug, random_state=None, max_iter=max_iter, n_workers=args.n_workers))
    pop_ratio_results.append(evaluation_func(policy.RatioPolicy, cases, name="ratio policy", population=True, debug=debug, random_state=None, max_iter=max_iter, n_workers=args.n_workers))
    pop_rand_results.append(evaluation_func(policy.RandomPolicy, cases, name="random", population=True, debug=debug, random_state=None, max_iter=max_iter, n_workers=args.n_workers))

end = time.time()
print("\n\nFinished experiments at %s (elapsed %0.2f seconds)" % (datetime.now(), end-start))

# Save results
results = [pop_mb_results, pop_ratio_results, pop_rand_results]
filename = save_results(results)
print("Saved to file \"%s\"" % filename)
