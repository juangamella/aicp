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
from src import evaluation, policy
import argparse

# --------------------------------------------------------------------
# Parse input parameters

# Definitions and default settings
arguments = {
    'n_workers': {'default': 1, 'type': int},
    'batch_size': {'default': 20000, 'type': int},
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
    'alpha': {'default': 0.01, 'type': float},
    'tag' : {'default': "", 'type': str}}

# Parse settings from input
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
print(args)

# --------------------------------------------------------------------
# Generate test cases

if args.max_iter == -1:
    max_iter = args.n_max
else:
    max_iter = args.max_iter
    
P = args.n_min if args.n_min == args.n_max else (args.n_min, args.n_max)

n_workers = None if args.n_workers == -1 else args.n_workers

# Generate test cases
cases = evaluation.gen_cases(args.G,
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
    policies = [policy.RandomPolicyF,
                policy.ProposedPolicyEF,
                policy.ProposedPolicyRF,
                policy.ProposedPolicyERF,
                policy.MarkovPolicyF,
                policy.ProposedPolicyMEF,
                policy.ProposedPolicyMRF,
                policy.ProposedPolicyMERF]
    names = ["random", "e", "r", "e + r", "markov", "markov + e", "markov + r", "markov + e + r"]
    
evaluation_params = {'population': population,
                     'n': args.n,
                     'alpha': args.alpha,
                     'debug': args.debug,
                     'random_state': None,
                     'max_iter': max_iter,
                     'n_workers': n_workers,
                     'batch_size': args.batch_size}

results = evaluation.evaluate_policies(cases, args.runs, policies, names, **evaluation_params)
end = time.time()
print("\n\nFinished experiments at %s (elapsed %0.2f seconds)" % (datetime.now(), end-start))

# --------------------------------------------------------------------
# Save results

filename = "experiments/results_%d" % end
for k,v in vars(args).items():
    filename = filename + "_" + k + ":" + str(v)
filename = filename + ".pickle"
pickle.dump([cases] + results, open(filename, "wb"))
print("Saved to file \"%s\"" % filename)
