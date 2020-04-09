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

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools

def matrix_block(M, rows, cols):
    """
    Select a block of a matrix given by the row and column indices
    """
    (n,m) = M.shape
    idx_rows = np.tile(np.array([rows]).T,len(cols)).flatten()
    idx_cols = np.tile(cols, (len(rows),1)).flatten()
    return M[idx_rows, idx_cols].reshape(len(rows), len(cols))

def sampling_matrix(W):
    """Given the weighted adjacency matrix of a DAG, return
    the matrix A such that the DAG generates samples
      A @ diag(var)^1/2 @ Z + mu
    where Z is an isotropic normal, and var/mu are the variances/means
    of the noise variables of the graph.
    """
    p = len(W)
    return np.linalg.inv(np.eye(p) - W.T)

def all_but(k,p):
    """Return [0,...,p-1] without k"""
    k = np.atleast_1d(k)
    return [i for i in range(p) if not i in k]

def combinations(p, target):
    """Return all possible subsets of the set {0...p-1} \ {target}"""
    base = set(range(p)) - {target}
    sets = []
    for size in range(p):
        sets += [set(s) for s in itertools.combinations(base, size)]
    return sets

def nonzero(A, tol=1e-12):
    """Return the indices of the nonzero (up to tol) elements in A"""
    return np.where(np.abs(A) > tol)[0]

def graph_info(i, W, interventions = None):
    """Returns the parents, children, parents of children and markov
    blanket of variable i in DAG W, using the graph structure
    """
    G = nx.from_numpy_matrix(W, create_using = nx.DiGraph)
    parents = set(G.predecessors(i))
    children = set(G.successors(i))
    parents_of_children = set()
    for child in children:
        parents_of_children.update(G.predecessors(child))
    if len(children) > 0: parents_of_children.remove(i)
    mb = parents.union(children, parents_of_children)
    return (parents, children, parents_of_children, mb)

def stable_blanket(i, W, interventions = set()):
    """Return the stable blanket using the graph structure"""
    G = nx.from_numpy_matrix(W, create_using = nx.DiGraph)
    parents = set(G.predecessors(i))
    children = set(G.successors(i))
    unstable_descendants = set()
    for j in interventions:
        if j in children:
            unstable_descendants.update({j})
            unstable_descendants.update(nx.algorithms.dag.descendants(G,j))
    stable_children = set.difference(children, unstable_descendants)
    parents_of_stable_children = set()
    for child in stable_children:
        parents_of_stable_children.update(G.predecessors(child))
    if len(stable_children) > 0: parents_of_stable_children.remove(i)
    sb = set.union(parents, stable_children, parents_of_stable_children)
    return sb

def descendants(i, W):
    """Return the descendants of a node using the graph structure"""
    G = nx.from_numpy_matrix(W, create_using = nx.DiGraph)
    return nx.algorithms.dag.descendants(G, i)

def ancestors(i, W):
    """Return the ancestors of a node using the graph structure"""
    G = nx.from_numpy_matrix(W, create_using = nx.DiGraph)
    return nx.algorithms.dag.ancestors(G, i)

def plot_graph(W, block=False):
    G = nx.from_numpy_matrix(W, create_using = nx.DiGraph)
    try:
        pos = nx.drawing.layout.planar_layout(G, scale=0.5)
    except nx.exception.NetworkXException:
        pos = nx.drawing.layout.shell_layout(G, scale=0.5)
    edge_labels = nx.get_edge_attributes(G,'weight')
    p = len(W)
    node_labels = dict(zip(np.arange(p), map(lambda i: "$X_{%d}$" %i, range(p))))
    # Plot
    fig = plt.figure()
    params = {'node_color': 'white',
              'edgecolors': 'black',
              'node_size': 900,
              'linewidths': 1.5,
              'width': 1.5,
              'arrowsize': 20,
              'arrowstyle': '->',
              'min_target_margin': 1000,
              'labels': node_labels}
    nx.draw(G,pos, **params)
    fig.set_facecolor("white")
    plt.show(block = block)

def allclose(A, B, rtol=1e-5, atol=1e-8):
    """Use np.allclose to compare, but relative tolerance is relative to
    the smallest element compared
    """
    return np.allclose(np.maximum(A,B), np.minimum(A,B), rtol, atol)
    
# Example graphs

def eg1():
    W = np.array([[0, 1, -1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    markov_blankets = [[1,2],
                       [0,3,2],
                       [0,3,1],
                       [1,2,4],
                       [3]]
    parents = [[],
               [0],
               [0],
               [1,2],
               [3]]
    return W, parents, markov_blankets

def eg2():
    W = np.array([[0, 1, -1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])
    markov_blankets = [[1,2],
                       [0,3,2],
                       [0,3,1],
                       [1,2,4,5],
                       [3,5],
                       [3,4]]
    parents = [[],
               [0],
               [0],
               [1,2],
               [3,5],
               []]
    return W, parents, markov_blankets

def eg3():
    W = np.array([[0, 1, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])
    markov_blankets = [[1,2],
                       [0,3,2],
                       [0,3,1],
                       [1,2,4,5],
                       [3,5,7],
                       [3,4,6],
                       [5],
                       [4]]
    parents = [[],
               [0],
               [0],
               [1,2],
               [3,5],
               [],
               [5],
               [4]]
    return W, parents, markov_blankets

def eg4():
    W = np.array([[0,0,1,0],
                  [0,0,1,0],
                  [0,0,0,1],
                  [0,0,0,0]])
    markov_blankets = [[1,2],
                       [0,2],
                       [0,1,3],
                       [2]]
    parents = [[],
               [],
               [0,1],
               [2]]
    return W, parents, markov_blankets

def eg5():
    W = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 1., 0., 1., 0., 0.],
                  [0., 1., 0., 0., 0., 1., 1., 0.],
                  [1., 0., 0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 1., 1., 0., 0.]])
    ordering = np.array([7, 4, 2, 3, 6, 5, 0, 1])
    parents = [[4,6,7],
               [0,2,3],
               [],
               [2],
               [7],
               [2,3,7],
               [3,4],
               []]
    children = [[1],
                [],
                [1,3,5],
                [1,5,6],
                [0,6],
                [],
                [0],
                [0,4,5]]
    markov_blankets = [[4,6,7,1,2,3],
                       [0,2,3],
                       [1,3,5,0,7],
                       [0,1,2,4,5,6,7],
                       [7,0,6,7,3],
                       [2,3,7],
                       [3,4,0,7],
                       [0,4,5,2,3,6]]
    return W, parents, markov_blankets

def eg6():
    W = np.array([[0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    parents = [[],
               [],
               [0,1],
               [2],
               [0,1,3]]
    children = [[2],
                [2],
                [3],
                [4],
                []]
    markov_blankets = [[1,2,3,4],
                       [0,2,3,4],
                       [0,1,3],
                       [2,4,0,1],
                       [0,1,3]]
    return W, parents, markov_blankets

