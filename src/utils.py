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

import numpy as np

def matrix_block(M, rows, cols):
    """
    Select a block of a matrix given by the row and column indices
    """
    (n,m) = M.shape
    idx_rows = np.tile(np.array([rows]).T,len(cols)).flatten()
    idx_cols = np.tile(cols, (len(rows),1)).flatten()
    return M[idx_rows, idx_cols].reshape(len(rows), len(cols))

def sampling_matrix(W, ordering):
    """Given the weighted adjacency matrix and ordering of a DAG, return
    the matrix A such that the DAG generates samples
      A @ diag(var)^1/2 @ Z + mu
    where Z is an isotropic normal, and var/mu are the variances/means
    of the noise variables of the graph.
    """
    p = len(W)
    A = np.eye(p)
    W = W + A # set diagonal of W to 1
    for i in range(p):
        A[i,:] = np.sum(W[:,[i]] * A, axis=0)
    return A

def all_but(k,p):
    """Return [0,...,p-1] without k"""
    return [i for i in range(p) if i != k]

def nonzero(A, tol=1e-12):
    return np.where(np.abs(A) > tol)[0]

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
    ordering = np.arange(5)
    return W, ordering, parents, markov_blankets

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
    ordering = np.arange(6)
    return W, ordering, parents, markov_blankets

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
    ordering = np.arange(8)
    return W, ordering, parents, markov_blankets

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
    ordering = np.arange(4)
    return W, ordering, parents, markov_blankets
