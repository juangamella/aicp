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
    'debug': {'default': True, 'type': bool},
    'k': {'default': 3, 'type': float},
    'G': {'default': 100, 'type': int},
    'runs': {'default': 1, 'type': int},
    'n_min': {'default': 12, 'type': int},
    'n_max': {'default': 12, 'type': int},
    'w_min': {'default': 0.5, 'type': float},
    'w_max': {'default': 1, 'type': float},
    'var_min': {'default': 0, 'type': float},
    'var_max': {'default': 1, 'type': float},
    'int_min': {'default': 0, 'type': float},
    'int_max': {'default': 1, 'type': float},
    'random_state': {'default': 42, 'type': int},
    'n': {'default': 1000, 'type': int},
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
print(args) # For debugging

# Generate dataset

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

def analyse(estimate, truth):
    subset = int(set.issubset(estimate, truth))
    superset = int(set.issuperset(estimate, truth))
    len_superset = len(estimate) if superset else 0
    intersection = len(estimate & truth)
    atleastone = int(intersection > 0)
    return np.array([subset, superset, len_superset, atleastone, intersection])

estimates = []
subsup = np.array([0, 0, 0, 0, 0])
for i,case in enumerate(cases):
    print("Case %d/%d" % (i, args.G), end="\r") if i % 10 == 0 and args.debug else None
    sample = case.sem.sample(args.n)
    estimate = set(policy.markov_blanket(sample, case.target, debug=False))
    estimates.append(estimate)
    subsup += analyse(estimate, case.truth)

print(subsup / args.G)
