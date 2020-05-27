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

"""This module is the interface for running experiments. It parses the
command line parameters, calls src.evaluation to generate test cases
and evaluate the experiments, and writes the results to a file.

Optionally it can save the generated cases following the structure
employed by ABCD, so the method can run on the same test cases.
"""

import pickle
import time
from datetime import datetime
from src import evaluation, policy, utils
import argparse
import os
import numpy as np
import sempler

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
    'runs': {'default': 1, 'type': int},
    'max_iter': {'default': -1, 'type': int},
    'random_state': {'default': 42, 'type': int},
    'tag' : {'type': str},
    'debug': {'default': False, 'type': bool},
    'save_dataset': {'type': str},
    'load_dataset': {'type': str},
    'abcd': {'type': bool, 'default': False}, # ABCD settings: Run only random, e + r, markov + e + r
    'G': {'default': 4, 'type': int},
    'k': {'default': 3, 'type': float},
    'p_min': {'default': 8, 'type': int},
    'p_max': {'default': 8, 'type': int},
    'w_min': {'default': 0.1, 'type': float},
    'w_max': {'default': 1, 'type': float},
    'var_min': {'default': 0, 'type': float},
    'var_max': {'default': 1, 'type': float},
    'int_min': {'default': 0, 'type': float},
    'int_max': {'default': 1, 'type': float},
    'do': {'default': False, 'type': bool},
    'i_mean': {'default': 10, 'type': float},
    'i_var': {'default': 1, 'type': float},
    'ot': {'type': int, 'default': 0},
    'finite': {'default': False, 'type': bool},
    'n': {'default': 100, 'type': int},
    'n_obs': {'type': int},
    'alpha': {'default': 0.01, 'type': float},
    'nsp': {'type': bool, 'default': False}
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
excluded_keys += ['nsp'] if not args.nsp else []

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
        sem = sempler.LGANM(W, variances[i], means[i])
        truth = utils.graph_info(targets[i], W)[0]
        cases.append(evaluation.TestCase(i, sem, targets[i], truth))
    excluded_keys += ['k', 'w_min', 'w_max', 'var_min', 'var_max', 'int_min', 'int_max', 'random_state', 'p_min', 'p_max']
# Or generate dataset
else:
    P = args.p_min if args.p_min == args.p_max else (args.p_min, args.p_max)
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
               'do',
               'i_mean',
               'i_var',
               'save_dataset',
               'load_dataset',
               'ot',
               'nsp',
               'abcd']
    exclude += ['tag'] if args.tag is None else []
    dir_name = args.save_dataset + "_%d" % time.time() + parameter_string(args, exclude)
    print("\nSaving test cases under %s/" % dir_name)
    # Save weighted adjacency matrix, means, variances and target
    for i,case in enumerate(cases):
        sem = case.sem
        os.makedirs(os.path.join(dir_name, 'dags', 'dag%d' % i), exist_ok=True)
        np.savetxt(os.path.join(dir_name, 'dags', 'dag%d' % i, 'adjacency.txt'), sem.W)
        np.savetxt(os.path.join(dir_name, 'dags', 'dag%d' % i, 'means.txt'), sem.means)
        np.savetxt(os.path.join(dir_name, 'dags', 'dag%d' % i, 'variances.txt'), sem.variances)
        np.savetxt(os.path.join(dir_name, 'dags', 'dag%d' % i, 'target.txt'), [case.target])    
else:

    # --------------------------------------------------------------------
    # Run experiments

    start = time.time()
    print("\n\nBeggining experiments on %d graphs at %s\n\n" % (len(cases), datetime.now()))

    population = not args.finite

    # Select which policies will be evaluated
    if population:
        # Note that in the population setting the empty-set strategy does
        # nothing, as variables are only intervened on once
        policies = [policy.PopRandom, policy.PopMarkov, policy.PopMarkovR] 
        names = ["random", "markov", "markov + r"]
        excluded_keys += ['n', 'n_obs', 'alpha']
    elif args.abcd:
        policies = [policy.Random,
                    policy.ER,
                    policy.E,
                    policy.R]
        names = ["random", "e + r", "e", "r"]
    else:
        policies = [policy.Random,
                    policy.E,
                    policy.R,
                    policy.ER,
                    policy.Markov,
                    policy.MarkovE,
                    policy.MarkovR,
                    policy.MarkovER]
        names = ["random", "e", "r", "e + r", "markov", "markov + e", "markov + r", "markov + e + r"]

    # Compose experimental parameters
    if args.max_iter == -1:
        max_iter = args.p_max
    else:
        max_iter = args.max_iter

    evaluation_params = {'population': population,
                         'debug': args.debug,
                         'max_iter': max_iter,
                         'intervention_type': 'do' if args.do else 'shift',
                         'intervention_mean': args.i_mean,
                         'intervention_var': args.i_var,
                         'off_targets': args.ot,
                         'speedup': not args.nsp,
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
