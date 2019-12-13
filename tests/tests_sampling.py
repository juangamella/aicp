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

#---------------------------------------------------------------------
# Unit testing for module sampling.py

import unittest

import numpy as np
import networkx as nx
from .context import src
from src.utils import sampling_matrix
from src.normal_distribution import NormalDistribution

# Tested functions
from src.sampling import dag_full, dag_avg_deg, LGSEM

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
        # Test the initialization of an LGSEM object
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
        # Test that intercepts are set correctly
        p = 10
        (W, ordering) = dag_avg_deg(p, p/4, 1, 1)
        intercepts = np.arange(p)
        sem = LGSEM(W, ordering, (0,1), intercepts = intercepts)
        self.assertTrue((sem.intercepts == intercepts).all())

    def test_sampling_args(self):
        variances = np.array([1,2,3])
        intercepts = np.array([3,4,5])
        W = np.eye(3)
        ordering = np.arange(3)
        sem = LGSEM(W, ordering, variances, intercepts)
        self.assertEqual(np.ndarray, type(sem.sample(n=1)))
        self.assertEqual(NormalDistribution, type(sem.sample(n=1, population=True)))
        self.assertEqual(NormalDistribution, type(sem.sample(population=True)))
        
    def test_sampling_1(self):
        # Test sampling of DAG with one variable
        np.random.seed(42)
        p = 1
        n = round(1e6)
        (W, ordering) = dag_full(p)
        sem = LGSEM(W, ordering, (1,1))
        # Observational data
        truth = np.random.normal(0,1,size=(n,1))
        samples = sem.sample(n)
        self.assertTrue(same_normal(truth, samples, atol=1e-1))
        # Under do intervention
        truth = np.ones((n,1))
        samples = sem.sample(n, do_interventions = np.array([[0, 1]]))
        self.assertTrue((truth == samples).all())
        # Under noise intervention
        truth = np.random.normal(1,2,size=(n,1))
        samples = sem.sample(n, noise_interventions = np.array([[0, 1, 4]]))
        self.assertTrue(same_normal(truth, samples, atol=1e-1))

    def test_sampling_2(self):
        # Test that the distribution of a 4 variable DAG with upper
        # triangular, all ones adj. matrix matches what we expect
        # using the path method
        p = 4
        n = round(1e6)
        (W, ordering) = dag_full(p)
        sem = LGSEM(W, ordering, (0.16,0.16))
        np.random.seed(42)
        noise = np.random.normal([0,0,0,0],[.4, .4, .4, .4], size=(n,4))
        truth = np.zeros((n,p))
        truth[:,0] = noise[:,0]
        truth[:,1] = noise[:,0] + noise[:,1]
        truth[:,2] = 2*noise[:,0] + noise[:,1] + noise[:,2]
        truth[:,3] = 4*noise[:,0] + 2*noise[:,1] + noise[:,2] + noise[:,3]
        samples = sem.sample(n)
        self.assertTrue(same_normal(truth, samples))
        
    def test_interventions_1(self):
        # Test sampling and interventions on a custom DAG, comparing
        # with results obtained via the path method
        np.random.seed(42)
        p = 6
        n = round(1e6)
        W = np.array([[0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        ordering = np.arange(p)
        sem = LGSEM(W, ordering, (0.16,0.16))

        # Test observational data
        M = np.array([[1, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [1, 0, 1, 1, 0, 0],
                      [2, 1, 1, 1, 1, 0],
                      [4, 2, 2, 2, 1, 1]])
        noise = np.random.normal(np.zeros(p), np.ones(p)*0.4, size=(n,p))
        truth = noise @ M.T
        samples = sem.sample(n)
        self.assertTrue(same_normal(truth, samples))
        
        # Test under do-interventions on X1
        noise = np.random.normal([2.1,0,0,0,0,0], [0,.4, .4, .4, .4, .4], size=(n,p))
        truth = noise @ M.T
        samples = sem.sample(n, do_interventions = np.array([[0,2.1]]))
        self.assertTrue(same_normal(truth, samples))
        
        # Test under do-intervention on X1 and noise interventions X2 and X5
        do_int = np.array([[0,2]])
        noise_int = np.array([[1, 2, 0.25], [4, 1, 0.25]])
        noise = np.random.normal([2,2,0,0,1,0], [0,.5,.4,.4,.5,.4], size=(n,p))
        M = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [1, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [1, 1, 1, 1, 1, 1]])
        truth = noise @ M.T
        samples = sem.sample(n, do_interventions=do_int, noise_interventions = noise_int)
        self.assertTrue(same_normal(truth, samples))

    def test_interventions_2(self):
        # Test that the means and variances of variables in the joint
        # distribution are what is expected via the path method
        W = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])
        ordering = np.arange(3)
        n = round(1e6)
        variances = np.array([1,2,3])*0.1
        intercepts = np.array([1,2,3])
        sem = LGSEM(W, ordering, variances, intercepts)
        np.random.seed(42)
        # Test observational data
        # Build truth
        noise = np.random.normal(intercepts, variances**0.5, size=(n,3))
        truth = np.zeros_like(noise)
        truth[:,0] = noise[:,0]
        truth[:,1] = truth[:,0]*W[0,1] + noise[:,1]
        truth[:,2] = truth[:,0]*W[0,2] + truth[:,1]*W[1,2] + noise[:,2]
        samples = sem.sample(n)
        self.assertTrue(same_normal(truth, samples))
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
        
        # Test under intervention on X1 <- N(0,0.1)
        variances = np.array([1., 1., 3.])*0.1
        intercepts = np.array([1., 0., 3.])
        noise = np.random.normal(intercepts, variances**0.5, size=(n,3))
        truth[:,0] = noise[:,0]
        truth[:,1] = noise[:,1]
        truth[:,2] = truth[:,0]*W[0,2] + truth[:,1]*W[1,2] + noise[:,2]
        samples = sem.sample(n, noise_interventions = np.array([[1,0,0.1]]))
        self.assertTrue(same_normal(truth, samples))
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
        variances = np.array([0., 2., 3.])*0.1
        intercepts = np.array([0., 2., 3.])
        noise = np.random.normal(intercepts, variances**0.5, size=(n,3))
        truth[:,0] = noise[:,0]
        truth[:,1] = truth[:,0]*W[0,1] + noise[:,1]
        truth[:,2] = truth[:,0]*W[0,2] + truth[:,1]*W[1,2] + noise[:,2]
        samples = sem.sample(n, do_interventions = np.array([[0,0]]))
        self.assertTrue(same_normal(truth, samples))
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

    def test_distribution(self):
        # Test "population" sampling
        W = np.array([[0,0,1,0],
                      [0,0,1,0],
                      [0,0,0,1],
                      [0,0,0,0]])
        ordering = np.arange(4)
        # Build SEM with unit weights and standard normal noise
        # variables
        sem = LGSEM(W, ordering, (1,1))
        # Observational Distribution
        distribution = sem.sample(population=True)
        true_cov = np.array([[1,0,1,1],
                             [0,1,1,1],
                             [1,1,3,3],
                             [1,1,3,4]])
        self.assertTrue((distribution.mean==np.zeros(4)).all())
        self.assertTrue((distribution.covariance==true_cov).all())
        # Do intervention on X1 <- 0
        distribution = sem.sample(population=True, do_interventions = np.array([[0,1]]))
        true_cov = np.array([[0,0,0,0],
                             [0,1,1,1],
                             [0,1,2,2],
                             [0,1,2,3]])
        self.assertTrue((distribution.mean==np.array([1,0,1,1])).all())
        self.assertTrue((distribution.covariance==true_cov).all())
        # Noise interventions on X1 <- N(0,2), X2 <- N(1,2)
        interventions = np.array([[0,0,2], [1,1,2]])
        distribution = sem.sample(population=True, noise_interventions=interventions)
        true_cov = np.array([[2,0,2,2],
                             [0,2,2,2],
                             [2,2,5,5],
                             [2,2,5,6]])
        self.assertTrue((distribution.mean==np.array([0,1,1,1])).all())
        self.assertTrue((distribution.covariance==true_cov).all())

    def test_sampling(self):
        # Test that sampling by building the multivariate normal
        # distribution works correctly (ie. NormalDistribution and sampling_matrix)
        P = [4,8,16]
        n = round(1e6)
        for p in P:
            (W, ordering) = dag_avg_deg(p, 3, -0.1, 0.1)
            variances = np.random.uniform(0.1, 0.2, p)
            intercepts = np.random.uniform(-1,1,p)
            # Generate samples through SEM
            X_sem = np.random.normal(intercepts, variances**0.5, size=(n,p))
            M = W + np.eye(p)
            for i in ordering:
                X_sem[:,i] = X_sem @ M[:,i]
            # Generate samples from Multivariate Normal
            A = sampling_matrix(W, ordering)
            distribution = NormalDistribution(A @ intercepts, A @ np.diag(variances) @ A.T)
            X_distr = distribution.sample(n)
            # Test
            self.assertTrue(same_normal(X_sem, X_distr))
        
def same_normal(sample_a, sample_b, atol=1e-2, debug=False):
    """
    Test (crudely, by L1 dist. of means and covariances) if samples
    from two distributions come from the same Gaussian
    """
    mean_a, mean_b = np.mean(sample_a, axis=0), np.mean(sample_b, axis=0)
    cov_a, cov_b = np.cov(sample_a, rowvar=False), np.cov(sample_b, rowvar=False)
    print("MEANS\n%s\n%s\n\nCOVARIANCES\n%s\n%s" % (mean_a, mean_b, cov_a, cov_b)) if debug else None
    means = np.allclose(mean_a, mean_b, atol=atol)
    covariances = np.allclose(cov_a, cov_b, atol=atol)
    return means and covariances


