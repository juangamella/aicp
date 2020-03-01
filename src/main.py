import networkx as nx
import matplotlib.pyplot as plt
from src.sampling import LGSEM
from src.icp import icp
import numpy as np
from src import utils
import time
from src import population_icp
from src import policy
from src import sampling
from sklearn import linear_model
from src.policy import Environments
# W = np.array([[0, 1, -1, 0, 0],
#               [0, 0, 0, 1, 0],
#               [0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 1],
#               [0, 0, 0, 0, 0]])

# # W = np.array([[0, 1, -1, 0, 0, 0],
# #               [0, 0, 0, 1, 0, 0],
# #               [0, 0, 0, 1, 0, 0],
# #               [0, 0, 0, 0, 1, 0],
# #               [0, 0, 0, 0, 0, 0],
# #               [0, 0, 0, 0, 1, 0]])
# ordering = np.arange(len(W))

#W = np.array([[0, 0, 0, 1],
              # [0, 0, 1, 1],
              # [0, 0, 0, 1],
              # [0, 0, 0, 0]])
#ordering = np.arange(4)g
W, ordering = sampling.dag_avg_deg(8, 3, 0.5, 1, random_state=51)
p = len(W)
#W = W * np.random.uniform(size=W.shape)
sem = LGSEM(W,ordering,(0,1))
n = 10000
e = sem.sample(n)
def ei(i):
    return sem.sample(n, do_interventions = np.array([[i,10,1]]))

# d = sem.sample(population = True)
# d0 = sem.sample(population = True, noise_interventions = np.array([[0,10,1]]))
# d1 = sem.sample(population = True, noise_interventions = np.array([[1,10,1]]))
# d2 = sem.sample(population = True, noise_interventions = np.array([[2,0,10]]))
# d3 = sem.sample(population = True, noise_interventions = np.array([[3,2,10]]))
# d4 = sem.sample(population = True, noise_interventions = np.array([[4,10,1]]))

# print(sem.W)
#_ = utils.plot_graph(sem.W)

target = 5

# print("finite sample icp")
# start = time.time()
# result = icp([e, ei(6)], target, alpha=0.01, max_predictors=None, selection = 'all', debug=True, stop_early=False)
# end = time.time()
# print("done in %0.2f seconds" % (end-start))

# print("population icp")
# start = time.time()
# result = population_icp.population_icp([d,d1], target, selection='all', debug=True)
# end = time.time()
# print("done in %0.2f seconds" % (end-start))

parents,_,_,mb = utils.graph_info(target, W)
print("Parents: %s" % parents)
print("Markov Blanket: %s" % mb)

# p = sem.p
# R = policy.ratios(p, result.accepted)
# print(R)
# for i,r in enumerate(R):
#     print("X%d = %d/%d=%0.4f" % (i, r*len(result.accepted), len(result.accepted), r))

# alphas = np.arange(1e-6, 1e-4, 1e-6)
# coefs = np.zeros((len(alphas), p))
# for i,a in enumerate(alphas):
#     predictors = utils.all_but(target, p)
#     coefs[i, predictors] = linear_model.Lasso(alpha=a, normalize=True).fit(e[:,predictors], e[:,target]).coef_
# for i in range(p):
#     plt.plot(alphas, coefs[:, i], label="x_%d" % i)
# plt.show(block=False)
# plt.legend()

# policy.markov_blanket(e, target, tol=0.15)
np.random.seed()
print("\nTesting lasso estimation")
correct = 0
superset = 0
contains_parents = 0
N = 100
i = 0
target = 1
np.random.seed()
begin = time.time()
while i < N:
    W, ordering = sampling.dag_avg_deg(8, 3, 0, 1)
    parents,_,_,mb = utils.graph_info(target, W)
    if len(parents) == 0:
        continue
    else:
        i += 1
    #print(i)
    sem = LGSEM(W,ordering,(0,1),(0,1))
    e = sem.sample(100)
    estimate = set(policy.markov_blanket(e, target, tol=1e-3))
    if mb == estimate:
        correct += 1
    if parents.issubset(estimate):
        contains_parents += 1
        if mb.issubset(estimate):
            superset += 1
print("correct %d/%d - contains parents %d/%d of which supersets %d/%d" % (correct, N, contains_parents, N, superset, N))
print(time.time()- begin)    
# # for i in np.argsort(result.mses):
# #     print("%s %0.5f" % (result.accepted[i], result.mses[i]))
# np.random.seed()
# # Test change in ratios with samples

# i = 7
# e = sem.sample(step)
# N = list(range(2, 1000, step))
# ratios = np.zeros((len(N),p))
# for j,n in enumerate(N):
#     ei = sem.sample(n, do_interventions = np.array([[i,10,0.01]]))
#     result = icp([e, ei], target, alpha=0.01, max_predictors=None, debug=False, stop_early=False)
#     ratios[j,:] = policy.ratios(p, result.accepted)
#     print("%s n=%d estimate=%s accepted sets:%d" % (ratios[j,:], n, result.estimate, len(result.accepted)))

# for k in range(p):
#     plt.plot(N, ratios[:,k], label="X_%d" % k) if k != target else None
# plt.plot([N[0], N[-1]], [0.5, 0.5], color="red")
# plt.legend()
# plt.title("Parents: %s intervention on %d" % (parents, i))
# plt.xlabel("No. of samples")
# plt.show(block=False)

# import pickle

# f = open("scratch/case_639.pickle", "rb")
# case = pickle.load(f)
# sem = case.sem
# d = sem.sample(population=True)
# d3 = sem.sample(population=True, noise_interventions = np.array([[3, 10, 2]]))
# #d10 = sem.sample(population=True, noise_interventions = np.array([[10, 10, 2]]))

# result = population_icp.population_icp([d, d3], case.target, debug=False)

# p = sem.p
# R = policy.ratios(p, result.accepted)
# print("Parents: %s" % case.truth)
# print(R)
# for i,r in enumerate(R):
#     print("X%d = %d/%d=%0.4f" % (i, r*len(result.accepted), len(result.accepted), r))


# def diff(A, B):
#     """Return elements in A that are not in B"""
#     return [i for i in A if i not in B]

# #(coefs, intercept) = d.regress(case.target, S)
