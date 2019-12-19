import networkx as nx
import matplotlib.pyplot as plt
from src.sampling import LGSEM
from src.icp import icp
import numpy as np

def ratios(p, accepted):
    one_hot = np.zeros((len(accepted), p))
    for i,s in enumerate(accepted):
        one_hot[i, list(s)] = 1
    return one_hot.sum(axis=0) / len(accepted)

# W = np.array([[0, 1, -1, 0, 0],
#               [0, 0, 0, 1, 0],
#               [0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 1],
#               [0, 0, 0, 0, 0]])
# ordering = np.arange(p)
# # W = np.array([[0, 1, -1, 0, 0, 0],
# #               [0, 0, 0, 1, 0, 0],
# #               [0, 0, 0, 1, 0, 0],
# #               [0, 0, 0, 0, 1, 0],
# #               [0, 0, 0, 0, 0, 0],
# #               [0, 0, 0, 0, 1, 0]])


# W, ordering, _, _ = utils.eg3()
# p = len(W)

# sem = LGSEM(W,ordering,(1,1))
# n = 1000000#round(1e4)
# e = sem.sample(n)
# e0 = sem.sample(n, noise_interventions = np.array([[0,10,1]]))
# e1 = sem.sample(n, noise_interventions = np.array([[1,10,1]]))
# e2 = sem.sample(n, do_interventions = np.array([[2,10,1]]))
# e4 = sem.sample(n, do_interventions = np.array([[4,10,1]]))
# #e5 = sem.sample(n, do_interventions = np.array([[5,10,1]]))
# print(sem.W)
# #_ = plot_graph(sem)

# environments = [e,e1]#, e2, e3, e4
# target = 3

# result = icp(environments, target, alpha=0.01, max_predictors=None, debug=True, stop_early=False)
# print("Intersection: %s" % result.estimate)


        
# R = ratios(p, result.accepted)
# for i,r in enumerate(R):
#     print("X%d = %d/%d=%0.2f" % (i, r*p, p, r))

# for i in np.argsort(result.mses):
#     print("%s %0.5f" % (result.accepted[i], result.mses[i]))



##--------------------------------------------------------------------

from src import sampling
from src import population_icp
from src import utils
from functools import reduce
y = 4
#W, ordering, _, _ = utils.eg3()
#W, ordering = sampling.dag_avg_deg(8,2.5,1,1,debug=True,random_state=2)
W, ordering = sampling.dag_avg_deg(8,2.5,1,1,debug=True,random_state=2)
# W = np.array([[0, 1, 1, 0],
#               [0, 0, 1, 0],
#               [0, 0, 0, 0],
#               [0, 0, 1, 0]])
# ordering = np.array([0, 1, 3, 2])
# W = np.array([[0, 1, 1],
#               [0, 0, 1],
#               [0, 0, 0]])
# ordering = np.array([0,1,2])

ordering = np.array([11, 9, 5, 4, 8, 1, 10, 2, 0, 3, 7, 6])

W = np.array([[0., 0., 0., 0., 0.,0., 0., 0.6493004 , 0., 0.,0., 0.],
              [0.16587633, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.],
              [0., 0., 0., 0.31913407, 0.,
               0., 0., 0., 0., 0.,
               0., 0.],
              [0., 0., 0., 0., 0.,
               0., 0.96663419, 0., 0., 0.,
               0., 0.],
              [0., 0., 0., 0.61031463, 0.,
               0., 0., 0., 0.14339083, 0.,
               0.46621716, 0.],
              [0., 0.12734835, 0., 0., 0.,
               0., 0., 0., 0., 0.,
               0., 0.],
              [0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0.,
               0., 0.],
              [0., 0., 0., 0., 0.,
               0., 0.64988289, 0., 0., 0.,
               0., 0.],
              [0., 0., 0.83605079, 0.59004019, 0.,
               0., 0.39212725, 0.10518279, 0., 0.,
               0., 0.],
              [0., 0., 0.88132803, 0.72336336, 0.97292084,
               0.70608919, 0., 0., 0.18451064, 0.,
               0., 0.],
              [0.4589405 , 0., 0., 0., 0.,
               0., 0., 0., 0., 0.,
               0., 0.],
              [0.70017783, 0.50926587, 0.53996192, 0., 0.,
               0.95663061, 0., 0.12697623, 0., 0.,
               0., 0.]])

#W = np.array([[0, 0, 0, 1, 0, 1],
              # [0, 0, 0, 1, 0, 1],
              # [0, 0, 0, 1, 0, 0],
              # [0, 0, 0, 0, 1, 0],
              # [0, 0, 0, 0, 0, 1],
              # [0, 0, 0, 0, 0, 0]])
ordering = np.arange(len(W))
W, ordering, _, _ = utils.eg6()

sem = sampling.LGSEM(W, ordering, (1,1))
utils.plot_graph(W)
p = len(W)
e = sem.sample(population=True)
v = 1.5

e0 = sem.sample(population=True, noise_interventions=np.array([[0,0,v]]))
e1 = sem.sample(population=True, noise_interventions=np.array([[1,0,v]]))
e2 = sem.sample(population=True, noise_interventions=np.array([[2,0,v]]))
e3 = sem.sample(population=True, noise_interventions=np.array([[3,0,v]]))
# e4 = sem.sample(population=True, noise_interventions=np.array([[4,0,v]]))
# e5 = sem.sample(population=True, noise_interventions=np.array([[5,0,v]]))
# e6 = sem.sample(population=True, noise_interventions=np.array([[6,0,v]]))
# e7 = sem.sample(population=True, noise_interventions=np.array([[7,0,v]]))
# e8 = sem.sample(population=True, noise_interventions=np.array([[8,0,v]]))
# e9 = sem.sample(population=True, noise_interventions=np.array([[9,0,v]]))
# e10 = sem.sample(population=True, noise_interventions=np.array([[10,0,v]]))
# e11 = sem.sample(population=True, noise_interventions=np.array([[11,0,v]]))


result = population_icp.population_icp([e, e2], y, debug=True, selection='all')
print(result.estimate)#, result.accepted, result.mses)
print("Stable blanket :", population_icp.stable_blanket(result.accepted, result.mses))
print("Nint:", reduce(lambda acc,x: acc.union(x), result.accepted))
R = ratios(len(W), result.accepted)
for i,r in enumerate(R):
    print("X%d = %d/%d=%0.2f" % (i, r*len(result.accepted), len(result.accepted), r))

# Children
#
# Parents
#
# PoC
#
