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

# --------------------------------------------------------------------
# Test cases

def gen_cases(n, p, k, w_min=1, w_max=1, var_min=1, var_max=1, int_min=0, int_max=0, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    cases = []
    for i in range(n):
        W, ordering = sampling.dag_avg_deg(p, k, w_min, w_max)
        sem = sampling.LGSEM(W, ordering, (var_min, var_max), (int_min, int_max))
        target = np.random.choice(range(p))
        (truth, _, _, _) = utils.graph_info(target, W)
        cases.append(policy.TestCase(sem, target, truth))
    return cases

# # Test case 1
# W, ordering = sampling.dag_avg_deg(8,2,1,2,random_state=2,debug=False)
# #utils.plot_graph(W, ordering)
# sem = sampling.LGSEM(W, ordering, (1,1))
# target = 3
# truth = {2}
# case_1 = policy.TestCase(sem, target, truth)

# cases = [case_1]

## Test case 2
# W = np.array([[0, 1, 1, 0],
#               [0, 0, 1, 0],
#               [0, 0, 0, 0],
#               [0, 0, 1, 0]])
# ordering = np.array([0, 1, 3, 2])
# sem = sampling.LGSEM(W, ordering, (1,1))
# target = 1
# truth = {0}
# case_2 = policy.TestCase(sem, target, truth)

cases = gen_cases(30, 16, 1.5)

# --------------------------------------------------------------------
# Evaluation

start = time.time()
print("\n\nBeggining experiments at %s\n\n" % datetime.now())

runs = 10

pop_rand_results = []
pop_mb_results = []

for i in range(runs):
    print("--- RUN %d ---" % i)
    pop_mb_results.append(policy.evaluate_policy(policy.MBPolicy, cases, name="markov blanket", population=True, debug=True, random_state=None))
    pop_rand_results.append(policy.evaluate_policy(policy.RandomPolicy, cases, name="random", population=True, debug=True, random_state=None))

end = time.time()
print("\n\nFinished experiments at %s (elapsed %0.2f seconds)" % (datetime.now(), end-start))

# Save results
results = [pop_rand_results, pop_mb_results]
filename = save_results(results)
print("Saved to file \"%s\"" % filename)


# --------------------------------------------------------------------
# Plotting

#results = load_results("experiments/results_1576347498.pickle")

results_mb = results[1]
results_rand = results[0]

no_ints_mb = np.zeros((10, 90))
no_ints_rand = np.zeros((10, 90))
correct_mb = np.zeros(90)
correct_rand = np.zeros(90)

for i in range(10):
    no_ints_mb[i,:] = list(map(lambda res: len(res.interventions()), results_mb[i]))
    no_ints_rand[i,:] = list(map(lambda res: len(res.interventions()), results_rand[i]))

for i in range(10):
    plt.scatter(np.arange(90), no_ints_mb[i,:], c='b', marker='+')
    plt.scatter(np.arange(90), no_ints_mb.mean(axis=0), c='b', marker='s')
    plt.scatter(np.arange(90), no_ints_rand[i,:], c='g', marker='x')
    plt.scatter(np.arange(90), no_ints_rand.mean(axis=0), c='b', marker='o')
plt.show(block=False)
