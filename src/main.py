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

"""
This module is just for scratch

TODO: Remove before publishing
"""

import networkx as nxi
import matplotlib.pyplot as plt
import sempler
from sempler import LGANM
from src.icp import icp, t_test, f_test
import numpy as np
from src import utils
import time
from src import population_icp
from src import policy
from src import sampling
from sklearn import linear_model

def diff(A, B):
    """Return elements in A that are not in B"""
    return [i for i in A if i not in B]

def regress(pooled, S, target):
    X = pooled[:, list(S)]
    Y = pooled[:,target]
    model = linear_model.LinearRegression(fit_intercept=True).fit(X, Y)
    coefs, intercept = model.coef_, model.intercept_
    res = Y - X @ coefs - intercept
    return (res, coefs, intercept)

def split_residuals(E, S, target, k=2, plot=False):
    residuals = regress(np.vstack(E), S, target)[0]
    split = np.random.choice(k, size=len(residuals))
    splitted_residuals = [residuals[split==i] for i in range(k)]
    if plot:
        plt.scatter(np.arange(len(residuals)), residuals, c=split)
        plt.plot([0,len(residuals)], [0,0], color='red')
        plt.show(block=False)
    return splitted_residuals

def test(ra, rb):
    return t_test(ra, rb), f_test(ra, rb)

W = sempler.dag_avg_deg(12, 3, 0.5, 1, random_state=50)
p = len(W)
#W = W * np.random.uniform(size=W.shape)
sem = LGANM(W,(0,1))
n = 100
e = sem.sample(n)

def ei(i, n_samples=n):
    return sem.sample(n_samples, do_interventions = {i: (10,1)})

# d = sem.sample(population = True)
# d0 = sem.sample(population = True, shift_interventions = {0: (10,1)})
# d1 = sem.sample(population = True, shift_interventions = {1: (10,1)})
# d2 = sem.sample(population = True, shift_interventions = {2: (0,10)})
# d3 = sem.sample(population = True, shift_interventions = {3: (2,10)})
# d4 = sem.sample(population = True, shift_interventions = {4: (10,1)})

# print(sem.W)
#_ = utils.plot_graph(sem.W)

target = 5

environments = [e, ei(4)] 

parents,children,pc,mb = utils.graph_info(target, W)
print("Parents: %s" % parents)
print("Children: %s" % children)
print("Parents of Children: %s" % pc)
print("Markov Blanket: %s" % mb)

print("finite sample icp")
start = time.time()
result = icp(environments, target, alpha=0.01, max_predictors=None, selection = 'all', debug=False, stop_early=False)
end = time.time()
print("done in %0.2f seconds" % (end-start))

# print("population icp")
# start = time.time()
# result = population_icp.population_icp([d,d1], target, selection='all', debug=True)
# end = time.time()
# print("done in %0.2f seconds" % (end-start))

print("Estimate: %s" % result.estimate)

p = sem.p
R = policy.ratios(p, result.accepted)
for i in range(p):
    print("X_%d = %0.4f" % (i,R[i]))
# for i,r in enumerate(R):
#     print("X%d = %d/%d=%0.4f" % (i, r*len(result.accepted), len(result.accepted), r))

# [ra, rb] = split_residuals(environments, {0}, target, k=2, plot=True)
# test(ra, rb)

