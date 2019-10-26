os.chdir('/home/juan/ETH/code_semester_project/src')
import networkx as nx
import matplotlib.pyplot as plt
def plot_graph(W):
    G = nx.from_numpy_matrix(W, create_using = nx.DiGraph)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    print("a: %s" % pos)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show()

from sampling import LGSEM
from icp import icp

W = np.array([[0, 1, 1, 1, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
p = len(W)
ordering = np.arange(p)
sem = LGSEM(W,ordering,(0.5,0.5))
n = round(1e5)
e = sem.sample(n)
e0 = sem.sample(n, noise_interventions = np.array([[0,5,1]]))
e1 = sem.sample(n, noise_interventions = np.array([[1,5,1]]))
e2 = sem.sample(n, do_interventions = np.array([[2,5,1]]))
e4 = sem.sample(n, do_interventions = np.array([[4,5,1]]))
print(sem.W)

#plot_graph(sem.W)

environments = [e, e0]#, e2, e3, e4
target = 3

S = icp(environments, target, alpha=0.05, max_predictors=None, debug=True)
print("S = %s" % S)
