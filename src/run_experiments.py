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

import pickle
import time
from datetime import datetime
from src import evaluation, policy, sampling, utils
import argparse
import os
import numpy as np

def parameter_string(args, excluded_keys):
    """Convert a Namespace object (from argparse) into a string, excluding
    some keys, to use as filename or dataset name"""
    string = ""
    for k,v in vars(args).items():
        value = str(v)
        value = value.replace('/', '')
        if k not in excluded_keys:
            string = string + "_" + k + ":" + value
    return string

# --------------------------------------------------------------------
# Parse input parameters

# Definitions and default settings
arguments = {
    'n_workers': {'default': 1, 'type': int},
    'batch_size': {'default': 20000, 'type': int},
    'debug': {'default': False, 'type': bool},
    'k': {'default': 3, 'type': float},
    'G': {'default': 4, 'type': int},
    'runs': {'default': 1, 'type': int},
    'n_min': {'default': 8, 'type': int},
    'n_max': {'default': 8, 'type': int},
    'w_min': {'default': 0, 'type': float},
    'w_max': {'default': 0.2, 'type': float},
    'var_min': {'default': 0, 'type': float},
    'var_max': {'default': 1, 'type': float},
    'int_min': {'default': 0, 'type': float},
    'int_max': {'default': 1, 'type': float},
    'i_mean': {'default': 10, 'type': float},
    'i_var': {'default': 1, 'type': float},
    'random_state': {'default': 42, 'type': int},
    'finite': {'default': False, 'type': bool},
    'max_iter': {'default': -1, 'type': int},
    'n': {'default': 100, 'type': int},
    'n_obs': {'type': int},
    'alpha': {'default': 0.01, 'type': float},
    'tag' : {'type': str},
    'save_dataset': {'type': str},
    'load_dataset': {'type': str},
    'abcd': {'type': bool, 'default': False}, # ABCD settings: Run only random, e + r, r
    'ot': {'type': int, 'default': 0},
    'sp': {'type': bool, 'default': False}
}


# Parse settings from input
parser = argparse.ArgumentParser(description='Run experiments')
for name, params in arguments.items():
    if params['type']==bool:
        options = {'action': 'store_true'}
    else:
        options = {'action': 'store', 'type': params['type']}
    if 'default' in params:
        options['default'] = params['default']
    parser.add_argument("--" + name,
                        dest=name,
                        required=False,
                        **options)

args = parser.parse_args()

# Parameters that will be excluded from the filename (see parameter_string function above)
excluded_keys = ['save_dataset', 'debug', 'n_workers', 'batch_size'] 
excluded_keys += ['tag'] if args.tag is None else []
excluded_keys += ['n_obs'] if args.n_obs is None else []
excluded_keys += ['abcd'] if not args.abcd else []
excluded_keys += ['ot'] if args.ot == 0 else []

print(args) # For debugging

# --------------------------------------------------------------------
# Generate (or load) test cases


# Load dataset
if args.load_dataset is not None:
    print("\nLoading test cases from %s" % args.load_dataset)
    # Load a dataset stored in the format used by ABCD
    G = len(os.listdir(os.path.join(args.load_dataset, 'dags')))
    Ws = [np.loadtxt(os.path.join(args.load_dataset, 'dags', 'dag%d' % i, 'adjacency.txt')) for i in range(G)]
    means = [np.loadtxt(os.path.join(args.load_dataset, 'dags', 'dag%d' % i, 'means.txt')) for i in range(G)]
    variances = [np.loadtxt(os.path.join(args.load_dataset, 'dags', 'dag%d' % i, 'variances.txt')) for i in range(G)]
    targets = [int(np.loadtxt(os.path.join(args.load_dataset, 'dags', 'dag%d' % i, 'target.txt'))) for i in range(G)]
    cases = []
    for i, W in enumerate(Ws):
        sem = sampling.LGSEM(W, variances[i], means[i])
        truth = utils.graph_info(targets[i], W)[0]
        cases.append(evaluation.TestCase(i, sem, targets[i], truth))
    excluded_keys += ['k', 'w_min', 'w_max', 'var_min', 'var_max', 'int_min', 'int_max', 'random_state', 'n_min', 'n_max']
# Or generate dataset
else:
    P = args.n_min if args.n_min == args.n_max else (args.n_min, args.n_max)
    cases = evaluation.gen_cases(args.G,
                                 P,
                                 args.k,
                                 args.w_min,
                                 args.w_max,
                                 args.var_min,
                                 args.var_max,
                                 args.int_min,
                                 args.int_max,
                                 args.random_state)
    excluded_keys += ['load_dataset']

# (Optionally) Save dataset according to format used by ABCD
if args.save_dataset is not None and args.load_dataset is None:
    # Compose directory name
    exclude = ['n_workers',
               'batch_size',
               'debug',
               'runs',
               'finite',
               'max_iter',
               'n',
               'n_obs',
               'alpha',
               'save_dataset',
               'load_dataset',
               'ot',
               'sp',
               'abcd']
    exclude += ['tag'] if args.tag is None else []
    dir_name = args.save_dataset + "_%d" % time.time() + parameter_string(args, exclude)
    print("\nSaving test cases under %s/" % dir_name)
    # Save weighted adjacency matrix, means, variances and target
    for i,case in enumerate(cases):
        sem = case.sem
        os.makedirs(os.path.join(dir_name, 'dags', 'dag%d' % i), exist_ok=True)
        np.savetxt(os.path.join(dir_name, 'dags', 'dag%d' % i, 'adjacency.txt'), sem.W)
        np.savetxt(os.path.join(dir_name, 'dags', 'dag%d' % i, 'means.txt'), sem.intercepts)
        np.savetxt(os.path.join(dir_name, 'dags', 'dag%d' % i, 'variances.txt'), sem.variances)
        np.savetxt(os.path.join(dir_name, 'dags', 'dag%d' % i, 'target.txt'), [case.target])

# --------------------------------------------------------------------
# Run experiments

start = time.time()
print("\n\nBeggining experiments on %d graphs at %s\n\n" % (len(cases), datetime.now()))

population = not args.finite

# Select which policies will be evaluated
if population:
    policies = [policy.MBPolicy, policy.RatioPolicy, policy.RandomPolicy]
    names = ["markov blanket", "ratio policy", "random"]
    excluded_keys += ['n', 'n_obs', 'alpha']
elif args.abcd:
    policies = [policy.RandomPolicyF,
                policy.ProposedPolicyERF,
                policy.ProposedPolicyRF]
    names = ["random", "e + r", "r"]
else:
    policies = [policy.RandomPolicyF,
                policy.ProposedPolicyEF,
                policy.ProposedPolicyRF,
                policy.ProposedPolicyERF,
                policy.MarkovPolicyF,
                policy.ProposedPolicyMEF,
                policy.ProposedPolicyMRF,
                policy.ProposedPolicyMERF]
    names = ["random", "e", "r", "e + r", "markov", "markov + e", "markov + r", "markov + e + r"]

# Compose experimental parameters
if args.max_iter == -1:
    max_iter = args.n_max
else:
    max_iter = args.max_iter
    
evaluation_params = {'population': population,
                     'debug': args.debug,
                     'max_iter': max_iter,
                     'intervention_mean': args.i_mean,
                     'intervention_var': args.i_var,
                     'off_targets': args.ot,
                     'speedup': args.sp,
                     'n_int': args.n,
                     'n_obs': args.n if args.n_obs is None else args.n_obs,
                     'alpha': args.alpha}

n_workers = None if args.n_workers == -1 else args.n_workers

results = evaluation.evaluate_policies(cases, policies, names, args.runs, evaluation_params, n_workers, args.batch_size)
end = time.time()
print("\n\nFinished experiments at %s (elapsed %0.2f seconds)" % (datetime.now(), end-start))

# --------------------------------------------------------------------
# Save results

# Compose filename
filename = "experiments/results_%d" % end
filename = filename + parameter_string(args, excluded_keys) + ".pickle"

# Pickle results
pickle.dump([cases] + results, open(filename, "wb"))
print("Saved to file \"%s\"" % filename)