import networkx as nx
import matplotlib.pyplot as plt
from src.sampling import LGSEM
from src.icp import icp
import numpy as np



W = np.array([[0, 1, -1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])

# W = np.array([[0, 1, -1, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 1, 0]])

p = len(W)
ordering = np.arange(p)
sem = LGSEM(W,ordering,(1,1))
n = round(1e4)
e = sem.sample(n)
e0 = sem.sample(n, noise_interventions = np.array([[0,10,1]]))
e1 = sem.sample(n, noise_interventions = np.array([[1,10,1]]))
e2 = sem.sample(n, do_interventions = np.array([[2,10,1]]))
e4 = sem.sample(n, do_interventions = np.array([[4,10,1]]))
#e5 = sem.sample(n, do_interventions = np.array([[5,10,1]]))
print(sem.W)
#_ = plot_graph(sem)

environments = [e, e1, e0]#, e2, e3, e4
target = 3

result = icp(environments, target, alpha=0.01, max_predictors=None, debug=True, stop_early=False)
print("Intersection: %s" % result.estimate)

def ratios(p, accepted):
    one_hot = np.zeros((len(accepted), p))
    for i,s in enumerate(accepted):
        one_hot[i, list(s)] = 1
    return one_hot.sum(axis=0) / len(accepted)
        
R = ratios(p, result.accepted)
for i,r in enumerate(R):
    print("X%d = %d/%d=%0.2f" % (i, r*p, p, r))

for i in np.argsort(result.mses):
    print("%s %0.5f" % (result.accepted[i], result.mses[i]))



##--------------------------------------------------------------------

from src import sampling
from src import population_icp
from src import utils
from functools import reduce
y = 3
#W, ordering, _, _ = utils.eg3()
W, ordering = sampling.dag_avg_deg(8,2.5,1,2,debug=True,random_state=2)
sem = sampling.LGSEM(W, ordering, (1,1))
e = sem.sample(population=True)
e_obs = sem.sample(n=round(1e6))
v = 2

e0 = sem.sample(population=True, noise_interventions=np.array([[0,0,v]]))
e1 = sem.sample(population=True, noise_interventions=np.array([[1,0,v]]))
e2 = sem.sample(population=True, noise_interventions=np.array([[2,1,v]]))
e3 = sem.sample(population=True, noise_interventions=np.array([[3,0,v]]))
e4 = sem.sample(population=True, noise_interventions=np.array([[4,0,v]]))
e5 = sem.sample(population=True, noise_interventions=np.array([[5,0,v]]))
e6 = sem.sample(population=True, noise_interventions=np.array([[6,0,v]]))
e7 = sem.sample(population=True, noise_interventions=np.array([[7,0,v]]))

e24 = sem.sample(population=True, noise_interventions=np.array([[2,0.3,v], [4, 0.3, v]]))

print()
print(population_icp.pooled_regression(y, [], [e0]))
print(population_icp.pooled_regression(y, [], [e1]))
print(population_icp.pooled_regression(y, [], [e2]))
print(population_icp.pooled_regression(y, [], [e3]))
print(population_icp.pooled_regression(y, [], [e4]))

print()
S = []
s = np.random.uniform(size=len(S))
print(e0.conditional(y, S, s).mean, e0.conditional(y, S, s).covariance)
print(e1.conditional(y, S, s).mean, e1.conditional(y, S, s).covariance)
print(e2.conditional(y, S, s).mean, e2.conditional(y, S, s).covariance)
print(e3.conditional(y, S, s).mean, e3.conditional(y, S, s).covariance)
print(e4.conditional(y, S, s).mean, e4.conditional(y, S, s).covariance)

result_pop = population_icp.population_icp([e, e2], y, debug=True, selection='all')
print(result_pop.estimate, result_pop.accepted, result_pop.mses)
print("Stable blanket :", population_icp.stable_blanket(result_pop.accepted, result_pop.mses))
print("Nint:", reduce(lambda acc,x: acc.union(x), result_pop.accepted))

#utils.plot_graph(W, ordering)
