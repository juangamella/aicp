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
    'seq': {'default': False, 'type': bool},
    'debug': {'default': False, 'type': bool},
    'avg_deg': {'default': 2, 'type': float},
    'G': {'default': 10, 'type': int},
    'runs': {'default': 1, 'type': int},
    'n_min': {'default': 8, 'type': int},
    'n_max': {'default': 8, 'type': int},
    'w_min': {'default': 1, 'type': float},
    'w_max': {'default': 2, 'type': float},
    'var_min': {'default': 0.1, 'type': float},
    'var_max': {'default': 1, 'type': float},
    'int_min': {'default': 0, 'type': float},
    'int_max': {'default': 1, 'type': float},
    'random_state': {'default': 42, 'type': int}}

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
args = argparse.Namespace(G=1000, avg_deg=4.0, debug=False, int_max=1, int_min=0, n_max=16, n_min=8, n_workers=4, random_state=2, runs=16, seq=False, var_max=1, var_min=0.1, w_max=3.0, w_min=1.5)
print()
print(args)
print()

# --------------------------------------------------------------------
# Run experiments
max_iter = args.n_max
debug = args.debug
use_parallelism = not args.seq
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

trouble = [0, 1, 2, 3, 5, 8, 9, 15, 18, 19, 22, 25, 27, 32, 33, 52, 56, 59, 61, 64, 67, 69, 72, 76, 78, 81, 89, 90, 97, 103, 112, 116, 120, 121, 122, 130, 135, 138, 140, 141, 143, 145, 148, 151, 154, 162, 164, 179, 182, 183, 190, 192, 193, 208, 210, 211, 214, 223, 224, 231, 233, 234, 236, 239, 244, 257, 262, 265, 266, 267, 268, 269, 273, 275, 280, 282, 286, 288, 290, 292, 297, 298, 300, 303, 331, 332, 340, 342, 345, 351, 362, 364, 372, 373, 381, 387, 390, 397, 400, 403, 408, 412, 417, 418, 425, 427, 429, 437, 439, 442, 447, 448, 450, 455, 463, 464, 466, 467, 475, 478, 479, 481, 486, 492, 497, 498, 499, 502, 508, 512, 514, 515, 520, 521, 523, 524, 527, 528, 529, 532, 535, 536, 538, 541, 544, 545, 547, 551, 556, 564, 571, 573, 578, 579, 581, 585, 587, 589, 590, 593, 597, 598, 605, 619, 632, 636, 638, 639, 654, 657, 662, 663, 664, 665, 667, 677, 679, 683, 686, 689, 694, 702, 709, 711, 713, 715, 718, 720, 731, 733, 735, 738, 739, 743, 747, 755, 756, 757, 758, 760, 761, 764, 768, 770, 773, 775, 779, 782, 788, 791, 792, 793, 798, 800, 801, 803, 805, 808, 809, 815, 817, 822, 827, 833, 834, 838, 846, 850, 851, 852, 853, 854, 855, 856, 864, 865, 867, 871, 880, 882, 884, 885, 887, 898, 899, 902, 911, 915, 932, 933, 934, 939, 940, 949, 950, 951, 952, 971, 972, 974, 978, 981, 982, 984, 993, 996, 997, 999]
cases = [cases[i] for i in trouble]

# Evaluate

start = time.time()
print("\n\nBeggining experiments on %d graphs at %s\n\n" % (len(cases), datetime.now()))

pop_rand_results = []
pop_mb_results = []
pop_ratio_results = []    

evaluation_params = {'population': True,
                     'debug': False,
                     'random_state': None,
                     'max_iter': max_iter}

if use_parallelism and __name__ == '__main__':
    print("Running in parallel")
    evaluation_func = evaluate_policy
    evaluation_params['n_workers'] = args.n_workers
elif use_parallelism:
    print("Not in __main__ module, running sequentially")
    evaluation_func = policy.evaluate_policy
else:
    print("Running sequentially")
    evaluation_func = policy.evaluate_policy

    
for i in range(args.runs):
    print("--- RUN %d ---" % i)
    pop_mb_results.append(evaluation_func(policy.MBPolicy, cases, name="markov blanket", **evaluation_params))
    pop_ratio_results.append(evaluation_func(policy.RatioPolicy, cases, name="ratio policy", **evaluation_params))
    pop_rand_results.append(evaluation_func(policy.RandomPolicy, cases, name="random", **evaluation_params))

end = time.time()
print("\n\nFinished experiments at %s (elapsed %0.2f seconds)" % (datetime.now(), end-start))

# Save results

filename = "experiments/results_%d" % end
for k,v in vars(args).items():
    filename = filename + "_" + k + ":" + str(v)
filename = filename + ".pickle"

results = [pop_mb_results, pop_ratio_results, pop_rand_results]
save_results(results, filename)
print("Saved to file \"%s\"" % filename)
