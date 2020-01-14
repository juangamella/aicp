import networkx as nx
import matplotlib.pyplot as plt
from src.sampling import LGSEM
from src.icp import icp
import numpy as np
from src import utils
import time
from src import population_icp
from src import policy

# # W = np.array([[0, 1, -1, 0, 0],
# #               [0, 0, 0, 1, 0],
# #               [0, 0, 0, 1, 0],
# #               [0, 0, 0, 0, 1],
# #               [0, 0, 0, 0, 0]])

# # # W = np.array([[0, 1, -1, 0, 0, 0],
# # #               [0, 0, 0, 1, 0, 0],
# # #               [0, 0, 0, 1, 0, 0],
# # #               [0, 0, 0, 0, 1, 0],
# # #               [0, 0, 0, 0, 0, 0],
# # #               [0, 0, 0, 0, 1, 0]])
# # ordering = np.arange(len(W))

# #W = np.array([[0, 0, 0, 1],
#               # [0, 0, 1, 1],
#               # [0, 0, 0, 1],
#               # [0, 0, 0, 0]])
# #ordering = np.arange(4)
# W, ordering, _, _ = utils.eg5()
# p = len(W)
# W = W * np.random.uniform(size=W.shape)
# sem = LGSEM(W,ordering,(1,1))
# n = round(1e4)
# e = sem.sample(n)
# e0 = sem.sample(n, noise_interventions = np.array([[0,10,1]]))
# e1 = sem.sample(n, noise_interventions = np.array([[1,10,1]]))
# e2 = sem.sample(n, do_interventions = np.array([[2,10,1]]))
# e4 = sem.sample(n, do_interventions = np.array([[4,10,1]]))

# d = sem.sample(population = True)
# d0 = sem.sample(population = True, noise_interventions = np.array([[0,10,1]]))
# d1 = sem.sample(population = True, noise_interventions = np.array([[1,10,1]]))
# d2 = sem.sample(population = True, noise_interventions = np.array([[2,0,10]]))
# d3 = sem.sample(population = True, noise_interventions = np.array([[3,0,10]]))
# d4 = sem.sample(population = True, noise_interventions = np.array([[4,10,1]]))

# print(sem.W)
# _ = utils.plot_graph(sem.W)

# target = 3

# print("finite sample icp")
# start = time.time()
# result = icp([e, e1], target, alpha=0.01, max_predictors=None, debug=True, stop_early=False)
# end = time.time()
# print("done in %0.2f seconds" % (end-start))

# # print("population icp")
# # start = time.time()
# # result = population_icp.population_icp([d,d3], target, selection='all', debug=False)
# # end = time.time()
# # print("done in %0.2f seconds" % (end-start))

# p = sem.p
# R = policy.ratios(p, result.accepted)
# print(R)
# for i,r in enumerate(R):
#     print("X%d = %d/%d=%0.4f" % (i, r*len(result.accepted), len(result.accepted), r))

# # for i in np.argsort(result.mses):
# #     print("%s %0.5f" % (result.accepted[i], result.mses[i]))


import pickle

f = open("scratch/case_381.pickle", "rb")
case = pickle.load(f)

sem = case.sem

d = sem.sample(population=True)
d11 = sem.sample(population=True, noise_interventions = np.array([[11, 0, 10]]))
d3 = sem.sample(population=True, noise_interventions = np.array([[3, 0, 10]]))

result = population_icp.population_icp([d, d11, d3], case.target, debug=False, atol=1e-2, rtol=1e-5)

p = sem.p
R = policy.ratios(p, result.accepted)
print("Parents: %s" % case.truth)
print(R)
for i,r in enumerate(R):
    print("X%d = %d/%d=%0.4f" % (i, r*len(result.accepted), len(result.accepted), r))


def diff(A, B):
    """Return elements in A that are not in B"""
    return [i for i in A if i not in B]

#(coefs, intercept) = d.regress(case.target, S)
