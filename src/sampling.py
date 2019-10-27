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
import matplotlib.pyplot as plt

#---------------------------------------------------------------------
# LGSEM class

class LGSEM:
    """
    Represents an Linear Gaussian SEM. Initialization randomly
    generates a new SEM. sample() generates observational and
    interventional samples from it
    """
    
    def __init__(self, W, ordering, variances, intercepts = None, debug=False):
        """Generate a random linear gaussian SEM, given
        - W: weight matrix
        - ordering: causal ordering of the variables
        - variances: either a vector of variances or a tuple
          indicating range for uniform sampling
        - intercepts: either a vector of intercepts, a tuple
          indicating the range for uniform sampling or None (zero
          intercepts)
        - graph_gen: the function used to generate the graph, by default
          "generate_dag_avg_dev"
        return a "SEM" object

        """
        self.W = W.copy()
        self.ordering = ordering.copy()
        self.p = len(W)

        # Set variances
        if isinstance(variances, tuple):
            self.variances = np.random.uniform(variances[0], variances[1], size=self.p)
        else:
            self.variances = variances.copy()
            
        # Set intercepts
        if intercepts is None:
            self.intercepts = np.zeros(self.p)
        elif isinstance(intercepts, tuple):
            self.intercepts = np.random.uniform(intercepts[0], intercepts[1], size=self.p)
        else:
            self.intercepts = intercepts.copy()
    
    def sample(self, n, do_interventions=None, noise_interventions=None, debug=False):
        """Generate n samples from a given Linear Gaussian SEM, under the given
        interventions (by default samples observational data)
        """
        # Must copy as they can be changed by intervention, but we
        # still want to keep the observational SEM
        W = self.W.copy()
        variances = self.variances.copy()
        intercepts = self.intercepts.copy()

        # Perform do interventions
        if do_interventions is not None:
            targets = do_interventions[:,0].astype(int)
            variances[targets] = 0
            intercepts[targets] = do_interventions[:,1]
            W[:,targets] = 0
            
        # Perform noise interventions
        if noise_interventions is not None:
            targets = noise_interventions[:,0].astype(int)
            intercepts[targets] = noise_interventions[:,1]
            variances[targets] = noise_interventions[:,2]
            W[:,targets] = 0
            
        # Sample
        # First fill X with the noise variables + intercepts
        print("Variances = %s, Intercepts = %s" % (variances, intercepts)) if debug else None
        X = np.random.normal(intercepts, variances**0.5, size=(n,self.p))
        M = W + np.eye(self.p)
        print(X) if debug else None
        for i in self.ordering:
            print("Sampling variable %d. Weights %s" % (i,M[:,i])) if debug else None
            X[:,i] = X @ M[:,i]
        return X

#---------------------------------------------------------------------
# DAG Generating Functions

def dag_avg_deg(p, k, w_min, w_max, debug=False):
    """
    Generate a random graph with p nodes and average degree k
    """
    # Generate adjacency matrix as if top. ordering is 1..p
    prob = k / (p-1)
    print("p = %d, k = %0.2f, P = %0.4f" % (p,k,prob)) if debug else None
    A = np.random.uniform(size = (p,p))
    A = (A <= prob).astype(float)
    A = np.triu(A, k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    
    # Permute rows/columns according to random topological ordering
    permutation = np.random.permutation(p)
    ordering = np.argsort(permutation)
    # Note the actual topological ordering is the "conjugate" of permutation eg. [3,1,2] -> [2,3,1]
    print("ordering = %s" % ordering) if debug else None
    print("avg degree = %0.2f" % (np.sum(A) * 2 / len(A))) if debug else None
    
    return (W[permutation, :][:, permutation], ordering)

def dag_full(p, w_min=1, w_max=1, debug=False):
    """Creates a fully connected DAG (ie. upper triangular adj. matrix
    with all ones) and causal ordering same as variable indices"""
    A = np.triu(np.ones((p,p)), k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    return (W, np.arange(p))

#---------------------------------------------------------------------
# Unit testing

import unittest
import networkx as nx
from scipy.stats import ttest_ind as ttest

# Tests for the DAG generation
class DAG_Tests(unittest.TestCase):
    def test_dag(self):
        for p in np.arange(2,100,5):
            (W, ordering) = dag_avg_deg(p, p/4, 0, 1)
            G = nx.from_numpy_matrix(W, create_using = nx.DiGraph)
            self.assertTrue(nx.is_directed_acyclic_graph(G))
            perm = np.argsort(ordering)

    def test_ordering(self):
        for p in np.arange(2,100,5):
            (W, ordering) = dag_avg_deg(p, p/4, 0, 1)
            B = W[ordering,:][:,ordering]
            self.assertTrue((B == np.triu(B,1)).all())

    def test_avg_degree(self):
        p = 1000
        for k in range(1,5):
            (W, _) = dag_avg_deg(p, k, 1, 2)
            av_deg = np.sum(W > 0) * 2 / p
            self.assertEqual(len(W), p)
            self.assertTrue(av_deg - k < 0.5)

    def test_disconnected_graph(self):
        (W, _) = dag_avg_deg(10, 0, 1, 1)
        self.assertEqual(np.sum(W), 0)

# Tests for the SEM generation and sampling
class SEM_Tests(unittest.TestCase):
    def test_basic(self):
        p = 10
        (W, ordering) = dag_avg_deg(p, p/4, 1, 1)
        sem = LGSEM(W, ordering, (1,1))
        self.assertTrue((sem.variances == np.ones(p)).all())
        self.assertTrue((sem.intercepts == np.zeros(p)).all())
        self.assertTrue(np.sum((sem.W == 0).astype(float) + (sem.W == 1).astype(float)), p*p)

    def test_memory(self):
        # Test that all arguments are copied and not simply stored by
        # reference
        variances = np.array([1,2,3])
        intercepts = np.array([3,4,5])
        W = np.eye(3)
        ordering = np.arange(3)
        sem = LGSEM(W, ordering, variances, intercepts)
        # Modify and compare
        variances[0] = 0
        intercepts[2] = 1
        W[0,0] = 2
        ordering[0] = 3
        self.assertFalse((W == sem.W).all())
        self.assertFalse((variances == sem.variances).all())
        self.assertFalse((intercepts == sem.intercepts).all())
        self.assertFalse((ordering == sem.ordering).all())
        
    def test_intercepts(self):
        p = 10
        (W, ordering) = dag_avg_deg(p, p/4, 1, 1)
        intercepts = np.arange(p)
        sem = LGSEM(W, ordering, (0,1), intercepts = intercepts)
        self.assertTrue((sem.intercepts == intercepts).all())

    def test_sampling_1(self):
        # Test sampling with one variable
        p = 1
        n = 100
        (W, ordering) = dag_full(p)
        sem = LGSEM(W, ordering, (1,1))
        # Observational data
        np.random.seed(42)
        truth = np.random.normal(0,1,size=(n,1))
        np.random.seed(42)
        samples = sem.sample(n)
        self.assertTrue((truth == samples).all())
        # Under do intervention
        truth = np.ones((n,1))
        samples = sem.sample(n, do_interventions = np.array([[0, 1]]))
        self.assertTrue((truth == samples).all())
        # Under noise intervention
        np.random.seed(42)
        truth = np.random.normal(1,2,size=(n,1))
        np.random.seed(42)
        samples = sem.sample(n, noise_interventions = np.array([[0, 1, 4]]))
        self.assertTrue((truth == samples).all())

    def test_sampling_2(self):
        # Test that the distribution of a 4 variable DAG with upper
        # triangular, all ones adj. matrix matches what we expect
        # using the path method
        p = 4
        n = 100
        (W, ordering) = dag_full(p)
        sem = LGSEM(W, ordering, (1,1))
        np.random.seed(42)
        noise = np.random.normal([0,0,0,0],[1,1,1,1], size=(n,4))
        truth = np.zeros((n,p))
        truth[:,0] = noise[:,0]
        truth[:,1] = noise[:,0] + noise[:,1]
        truth[:,2] = 2*noise[:,0] + noise[:,1] + noise[:,2]
        truth[:,3] = 4*noise[:,0] + 2*noise[:,1] + noise[:,2] + noise[:,3]
        np.random.seed(42)
        samples = sem.sample(n)
        self.assertTrue(np.allclose(truth, samples))
        
    def test_interventions_1(self):
        # Test sampling and interventions on a custom DAG, comparing
        # with results obtained via the path method
        p = 6
        n = 25
        W = np.array([[0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        ordering = np.arange(p)
        sem = LGSEM(W, ordering, (1,1))

        # Test observational data
        M = np.array([[1, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [1, 0, 1, 1, 0, 0],
                      [2, 1, 1, 1, 1, 0],
                      [4, 2, 2, 2, 1, 1]])
        np.random.seed(42)
        noise = np.random.normal(np.zeros(p), np.ones(p), size=(n,p))
        truth = noise @ M.T
        np.random.seed(42)
        samples = sem.sample(n)
        self.assertTrue(np.allclose(truth, samples))
        
        # Test under do-interventions on X1
        np.random.seed(42)
        noise = np.random.normal([2.1,0,0,0,0,0], [0,1,1,1,1,1], size=(n,p))
        truth = noise @ M.T
        np.random.seed(42)
        samples = sem.sample(n, do_interventions = np.array([[0,2.1]]))
        self.assertTrue(np.allclose(truth, samples))
        
        # Test under do-intervention on X1 and noise interventions X2 and X5
        do_int = np.array([[0,2]])
        noise_int = np.array([[1, 2, 1], [4, 1, 4]])
        np.random.seed(42)
        noise = np.random.normal([2,2,0,0,1,0], [0,1,1,1,2,1], size=(n,p))
        M = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [1, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [1, 1, 1, 1, 1, 1]])
        truth = noise @ M.T
        np.random.seed(42)
        samples = sem.sample(n, do_interventions=do_int, noise_interventions = noise_int)
        self.assertTrue(np.allclose(truth, samples))

    def test_interventions_2(self):
        W = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])
        ordering = np.arange(3)
        n = 100000
        variances = np.array([1,2,3])
        intercepts = np.array([1,2,3])
        sem = LGSEM(W, ordering, variances, intercepts)

        # Test observational data
        # Build truth
        np.random.seed(42)
        noise = np.random.normal(intercepts, variances**0.5, size=(n,3))
        truth = np.zeros_like(noise)
        truth[:,0] = noise[:,0]
        truth[:,1] = truth[:,0]*W[0,1] + noise[:,1]
        truth[:,2] = truth[:,0]*W[0,2] + truth[:,1]*W[1,2] + noise[:,2]
        np.random.seed(42)
        samples = sem.sample(n)
        self.assertTrue(np.allclose(truth, samples))
        # Test that variances/means are as expected
        true_vars, true_means = np.zeros(3), np.zeros(3)
        true_vars[0] = variances[0]
        true_vars[1] = W[0,1]**2 * variances[0] + variances[1]
        true_vars[2] = (W[0,1]*W[1,2] + W[0,2])**2 * variances[0] + W[1,2]**2 * variances[1] + variances[2]
        true_means[0] = intercepts[0]
        true_means[1] = W[0,1] * intercepts[0] + intercepts[1]
        true_means[2] = (W[0,1] * W[1,2] + W[0,2]) * intercepts[0] + W[1,2] * intercepts[1] + intercepts[2]
        self.assertTrue(np.allclose(true_vars, np.var(samples, axis=0), atol=1e-2))
        self.assertTrue(np.allclose(true_means, np.mean(samples, axis=0), atol=1e-2))
        
        # Test under intervention on X1 <- N(0,1)
        np.random.seed(42)
        variances = np.array([1., 1., 3.])
        intercepts = np.array([1., 0., 3.])
        noise = np.random.normal(intercepts, variances**0.5, size=(n,3))
        truth[:,0] = noise[:,0]
        truth[:,1] = noise[:,1]
        truth[:,2] = truth[:,0]*W[0,2] + truth[:,1]*W[1,2] + noise[:,2]
        np.random.seed(42)
        samples = sem.sample(n, noise_interventions = np.array([[1,0,1]]))
        self.assertTrue(np.allclose(truth, samples))
        # Test that variances/means are as expected
        true_vars, true_means = np.zeros(3), np.zeros(3)
        true_vars[0] = variances[0]
        true_vars[1] = variances[1]
        true_vars[2] = W[0,2]**2 * variances[0] + W[1,2]**2 * variances[1] + variances[2]
        true_means[0] = intercepts[0]
        true_means[1] = intercepts[1]
        true_means[2] = W[0,2] * intercepts[0] + W[1,2] * intercepts[1] + intercepts[2]
        self.assertTrue(np.allclose(true_vars, np.var(samples, axis=0), atol=1e-2))
        self.assertTrue(np.allclose(true_means, np.mean(samples, axis=0), atol=1e-2))
        
        # Test under intervention on do(X0 = 0)
        variances = np.array([0., 2., 3.])
        intercepts = np.array([0., 2., 3.])
        np.random.seed(42)
        noise = np.random.normal(intercepts, variances**0.5, size=(n,3))
        truth[:,0] = noise[:,0]
        truth[:,1] = truth[:,0]*W[0,1] + noise[:,1]
        truth[:,2] = truth[:,0]*W[0,2] + truth[:,1]*W[1,2] + noise[:,2]
        np.random.seed(42)
        samples = sem.sample(n, do_interventions = np.array([[0,0]]))
        self.assertTrue(np.allclose(truth, samples))
        # Test that variances/means are as expected
        true_vars, true_means = np.zeros(3), np.zeros(3)
        true_vars[0] = variances[0]
        true_vars[1] = W[0,1]**2 * variances[0] + variances[1]
        true_vars[2] = (W[0,1]*W[1,2] + W[0,2])**2 * variances[0] + W[1,2]**2 * variances[1] + variances[2]
        true_means[0] = intercepts[0]
        true_means[1] = W[0,1] * intercepts[0] + intercepts[1]
        true_means[2] = (W[0,1] * W[1,2] + W[0,2]) * intercepts[0] + W[1,2] * intercepts[1] + intercepts[2]
        self.assertTrue(np.allclose(true_vars, np.var(samples, axis=0), atol=1e-2))
        self.assertTrue(np.allclose(true_means, np.mean(samples, axis=0), atol=1e-2))
