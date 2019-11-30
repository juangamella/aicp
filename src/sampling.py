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
from utils import matrix_block

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
    
    def sample(self, n=round(1e5), population=False, do_interventions=None, noise_interventions=None, debug=False):
        """
        If population is set to False:
          - Generate n samples from a given Linear Gaussian SEM, under the given
            interventions (by default samples observational data)
        if set to True:
          - Return the "symbolic" joint distribution under the given
            interventions (see class NormalDistribution)
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
            
        # Sampling by building the joint distribution
        A = sampling_matrix(W, self.ordering)
        mean = A @ intercepts
        covariance = A @ np.diag(variances) @ A.T
        distribution = NormalDistribution(mean, covariance)
        if not population:
            return distribution.sample(n)
        else:
            return distribution

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

#---------------------------------------------------------------------
# NormalDistribution class

class NormalDistribution():
    """Symbolic representation of a normal distribution that allows for
    marginalization, conditioning and sampling
    
    Attributes:
      - mean: mean vector
      - covariance: covariance matrix
      - p: number of variables

    """
    def __init__(self, mean, covariance):
        self.p = len(mean)
        self.mean = mean.copy()
        self.covariance = covariance.copy()

    def sample(self, n):
        """Sample from the distribution"""
        return np.random.multivariate_normal(self.mean, self.covariance, size=n)

    def marginal(self, X):
        """Return the marginal distribution of the variables with indices X"""
        mean = self.mean[X].copy()
        covariance = matrix_block(self.covariance, X, X).copy()
        return NormalDistribution(mean, covariance)

    def conditional(self, Y, X, x):
        """Return the conditional distribution of the variables with indices Y
        given observations x of the variables with indices X

        """
        # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        cov_y = matrix_block(self.covariance, Y, Y)
        cov_x = matrix_block(self.covariance, X, X)
        cov_yx = matrix_block(self.covariance, Y, X)
        cov_xy = matrix_block(self.covariance, X, Y)
        mean_y = self.mean[Y]
        mean_x = self.mean[X]
        mean = mean_y + cov_yx @ np.linalg.inv(cov_x) @ (x - mean_x)
        covariance = cov_y - cov_yx @ np.linalg.inv(cov_x) @ cov_xy
        return NormalDistribution(mean,covariance)

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

    def test_sampling_matrix_fun(self):
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

class NormalDistributionTests(unittest.TestCase):
    def test_initialization(self):
        # Test initialization and that arguments are copied (not
        # stored by reference)
        covariance = np.array([[11,12,13],
                               [21,22,23],
                               [31,32,33]])
        mean = np.array([1,2,3])
        distribution = NormalDistribution(mean, covariance)
        self.assertTrue((mean == distribution.mean).all())
        self.assertTrue((covariance == distribution.covariance).all())
        mean[2] = 0
        covariance[:,2] = 0
        self.assertTrue(not (mean == distribution.mean).all())
        self.assertTrue(not (covariance == distribution.covariance).all())

    def test_marginalization(self):
        # Test marginalization logic
        covariance = np.array([[11,12,13],
                               [21,22,23],
                               [31,32,33]])
        mean = np.array([1,2,3])
        distribution = NormalDistribution(mean, covariance)
        # Marginalize X1
        marginal = distribution.marginal([0])
        self.assertTrue((marginal.mean == np.array([1])).all())
        self.assertTrue((marginal.covariance == np.array([11])).all())
        # Marginalize X1 and X3
        marginal = distribution.marginal([0,2])
        self.assertTrue((marginal.mean == np.array([1, 3])).all())
        self.assertTrue((marginal.covariance == np.array([[11, 13], [31, 33]])).all())

    def test_conditioning_1(self):
        # Test conditioning
        covariance = np.eye(3)
        mean = np.zeros(3)
        distribution = NormalDistribution(mean, covariance)
        # Condition X1 on X2=1 should yield the marginal of X1 (as they are independent)
        conditional = distribution.conditional([0], [1], np.array([1]))
        self.assertTrue((conditional.mean == np.array([0])).all())
        self.assertTrue((conditional.covariance == np.array([1])).all())
        self.assertTrue((conditional.mean == distribution.marginal([0]).mean).all())
        self.assertTrue((conditional.covariance == distribution.marginal([0]).covariance).all())
        # Conditioning X1, X2 on X3 = 1 should yield marginal of X1, X2
        conditional = distribution.conditional([0,1], [2], 0)
        self.assertTrue((conditional.mean == np.array([0, 0])).all())
        self.assertTrue((conditional.covariance == np.eye(2)).all())
        self.assertTrue((conditional.mean == distribution.marginal([0,1]).mean).all())
        self.assertTrue((conditional.covariance == distribution.marginal([0,1]).covariance).all())
        # Conditioning X1 on X2=2, X3 = 1 should yield marginal of X1, X2
        conditional = distribution.conditional([0], [1,2], np.array([2,1]))
        self.assertTrue((conditional.mean == np.array([0])).all())
        self.assertTrue((conditional.covariance == np.eye(1)).all())
        self.assertTrue((conditional.mean == distribution.marginal([0]).mean).all())
        self.assertTrue((conditional.covariance == distribution.marginal([0]).covariance).all())

    def test_conditioning_2(self):
        covariance = np.array([[1,2,3], [2,4,6], [3,6,9]]) # rank 1
        mean = np.array([1,2,3])
        dist = NormalDistribution(mean, covariance)
        # Conditioning X1 on X2 = 1 should give mean = 1/2 and variance = 0
        self.assertTrue((dist.conditional([0], [1], 1).mean == np.array([0.5])).all())
        self.assertTrue((dist.conditional([0], [1], 1).covariance == np.array([[0]])).all())
        # Conditioning X1 on X2 = 0 should give mean = 0 and variance = 0
        self.assertTrue((dist.conditional([0], [1], 0).mean == np.array([0])).all())
        self.assertTrue((dist.conditional([0], [1], 0.5).covariance == np.array([[0]])).all())

    def test_conditioning_3(self):
        covariance = np.array([[1, 0, 1, 1],
                               [0, 1, 1, 1],
                               [1, 1, 3, 3],
                               [1, 1, 3, 4]])
        mean = np.array([0,0,0,0])
        dist = NormalDistribution(mean, covariance)
        # Conditioning X3 on X1 = 1, X2 = 2 should give mean 3 and variance 1
        conditional = dist.conditional([2], [0,1], np.array([1,2]))
        self.assertTrue((conditional.mean == np.array([3])).all())
        self.assertTrue((conditional.covariance == np.array([[1]])).all())
        # Conditioning X4 on X1 = 1, X2 = 2 should give mean 3 and variance 2
        conditional = dist.conditional([3], [0,1], np.array([1,2]))
        self.assertTrue((conditional.mean == np.array([3])).all())
        self.assertTrue((conditional.covariance == np.array([[2]])).all())
        # Conditioning X3, X4 on X1 = 1, X2 = 2
        conditional = dist.conditional([2, 3], [0,1], np.array([1,2]))
        self.assertTrue((conditional.mean == np.array([3,3])).all())
        self.assertTrue((conditional.covariance == np.array([[1,1], [1,2]])).all())
        # Conditioning X3 on X1 = 1, X4 = 1 should give mean 1 and variance 3-7/3
        conditional = dist.conditional([2], [0,3], np.array([1,1]))
        self.assertTrue(np.allclose(conditional.mean, np.array([1])))
        self.assertTrue(np.allclose(conditional.covariance, np.array([[3-7/3]])))
        
    def test_sampling(self):
        # Test sampling of the distribution
        np.random.seed(42)
        n = round(1e6)
        covariance = 1e-2 * np.array([[1, 0, 1, 1],
                                      [0, 1, 1, 1],
                                      [1, 1, 3, 3],
                                      [1, 1, 3, 4]])
        mean = np.array([0,0,0,0])
        dist = NormalDistribution(mean, covariance)
        samples = dist.sample(n)
        self.assertTrue(np.allclose(np.mean(samples, axis=0), mean, atol=1e-1))
        self.assertTrue(np.allclose(np.cov(samples, rowvar=False), covariance,  atol=1e-1))


