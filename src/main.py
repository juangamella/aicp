os.chdir('/home/juan/ETH/code_semester_project/src')
import networkx as nx
import matplotlib.pyplot as plt
def plot_graph(sem):
    G = nx.from_numpy_matrix(sem.W, create_using = nx.DiGraph)
    pos = {}
    for i,node in enumerate(sem.ordering):
        x = (-1.1)**i
        y = -i*0.5
        pos[node] = (x,y)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show(block = False)

from sampling import LGSEM
from icp import icp
import numpy as np

W = np.array([[0, 1, -1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
p = len(W)
ordering = np.arange(p)
sem = LGSEM(W,ordering,(1,1))
n = round(1e5)
e = sem.sample(n)
e0 = sem.sample(n, noise_interventions = np.array([[0,5,1]]))
e1 = sem.sample(n, noise_interventions = np.array([[1,5,1]]))
e2 = sem.sample(n, do_interventions = np.array([[2,5,1]]))
e4 = sem.sample(n, do_interventions = np.array([[4,5,1]]))
print(sem.W)
#_ = plot_graph(sem)

environments = [e, e1, e4]#, e2, e3, e4
target = 3

(S, accepted) = icp(environments, target, alpha=0.01, max_predictors=None, debug=True, stop_early=False)
S,accepted

from icp import Data
from icp import confidence_intervals

data = Data(environments, target)
confidence_intervals(set([1,2]), np.array([0, 1, 1, 0, 0, 0]), data, 0.01)
