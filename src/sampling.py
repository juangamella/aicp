import numpy as np
import matplotlib.pyplot as plt

# DAG Generating Functions

def dag_default(p, debug=False):
    return dag_avg_deg(p, debug)

def dag_avg_deg(p, k=3, debug=False):
    """
    Generate a random graph with p nodes and average degree k
    """
    # Generate adjacency matrix as if top. ordering is 1..p
    prob = k / (p-1)
    print("p = %d, k = %0.2f, P = %0.4f" % (p,k,prob)) if debug else None
    A = np.random.uniform(size = (p,p))
    A = (A <= prob).astype(float)
    A = np.triu(A, k=1)
    
    # Permute rows/columns according to random topological ordering
    permutation = np.random.permutation(p)
    ordering = np.argsort(permutation)
    # Note the actual topological ordering is the "conjugate" of permutation eg. [3,1,2] -> [2,3,1]
    print("ordering = %s" % ordering) if debug else None
    print("avg degree = %0.2f" % (np.sum(A) * 2 / len(A))) if debug else None
    
    return (A[permutation, :][:, permutation], ordering)

def dag_full(p, debug=False):
    """Creates a fully connected DAG (ie. upper triangular adj. matrix
    with all ones) and causal ordering same as variable indices"""
    return (np.triu(np.ones((p,p)), k=1), np.arange(p))

def dag_custom(A, ordering):
    """Returns a generator for the DAG specified by the adjacency matrix and ordering"""
    gen = lambda p, debug=False: (A, ordering)
    return gen

# LGSEM class

class LGSEM:
    """
    Represents an Linear Gaussian SEM. Initialization randomly
    generates a new SEM. sample() generates observational and
    interventional samples from it
    """
    
    def __init__(self, p, w_min, w_max, var_min, var_max, intercepts = None, debug=False, graph_gen=dag_default):
        """
        Generate a random linear gaussian SEM, given
        - p: the number of variables in the SEM
        - w_min, w_max: lower and upper bounds for sampling weights
        - var_min, var_max: lower and upper bounds for sampling variances
        - intercepts: the "base value" of each variable (0 by default)
        - graph_gen: the function used to generate the graph, by default
          "generate_dag_avg_dev"
        return a "SEM" object
        """
        self.p = p
        
        # Generate DAG
        (A, ordering) = graph_gen(p, debug=debug)
        self.ordering = ordering
        
        # Sample weights and update adjacency matrix
        weights = np.random.uniform(w_min, w_max, size=A.shape)
        self.W = A * weights
        
        # Sample variances for noise
        self.variances = np.random.uniform(var_min, var_max, size=len(A))
        
        # Intercepts
        if intercepts is None:
            intercepts = np.zeros(len(A))
        self.intercepts = intercepts
    
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
            targets = do_interventions[:,0]
            variances[targets] = 0
            intercepts[targets] = do_interventions[:,1]
            W[:,targets] = 0
            
        # Perform noise interventions
        if noise_interventions is not None:
            targets = noise_interventions[:,0]
            intercepts[targets] = noise_interventions[:,1]
            variances[targets] = noise_interventions[:,2]
            W[:,targets] = 0
            
        # Sample
        # First fill X with the noise variables + intercepts
        print("Variances = %s, Intercepts = %s" % (variances, intercepts)) if debug else None
        X = np.random.normal(intercepts, variances**0.5, size=(n,self.p))
        M = W + np.eye(self.p)
        for i in self.ordering:
            print("Sampling variable %d. Weights %s" % (i,M[:,i])) if debug else None
            X[:,i] = X @ M[:,i]
            
        return X

# Unit testing

import unittest
import networkx as nx
from scipy.stats import ttest_ind as ttest

# Tests for the DAG generation
class DAG_Tests(unittest.TestCase):
    def test_dag(self):
        for p in np.arange(2,100,5):
            (A, ordering) = dag_avg_deg(p, p/4)
            G = nx.from_numpy_matrix(A, create_using = nx.DiGraph)
            self.assertTrue(nx.is_directed_acyclic_graph(G))
            perm = np.argsort(ordering)

    def test_ordering(self):
        for p in np.arange(2,100,5):
            (A, ordering) = dag_avg_deg(p, p/4)
            B = A[ordering,:][:,ordering]
            self.assertTrue((B == np.triu(B,1)).all())

    def test_avg_degree(self):
        p = 1000
        for k in range(1,5):
            (A, _) = dag_avg_deg(p, k)
            av_deg = np.sum(A) * 2 / p
            self.assertEqual(len(A), p)
            self.assertTrue(av_deg - k < 0.5)

    def test_disconnected_graph(self):
        (A, _) = dag_avg_deg(10, 0)
        self.assertEqual(np.sum(A), 0)

# Tests for the SEM generation and sampling
class SEM_Tests(unittest.TestCase):
    def test_basic(self):
        p = 10
        sem = LGSEM(p,1,1,1,1)
        self.assertTrue((sem.variances == np.ones(p)).all())
        self.assertTrue((sem.intercepts == np.zeros(p)).all())
        self.assertTrue(np.sum((sem.W == 0).astype(float) + (sem.W == 1).astype(float)), p*p)

    def test_intercepts(self):
        p = 10
        intercepts = np.arange(p)
        sem = LGSEM(p,0,1,0,1, intercepts = intercepts)
        self.assertTrue((sem.intercepts == intercepts).all())

    def test_sampling_1(self):
        # Test sampling with one variable
        p = 1
        n = 100
        sem = LGSEM(p,1,1,1,1, graph_gen=dag_full)
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
        sem = LGSEM(p,1,1,1,1, graph_gen=dag_full)
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
        A = np.array([[0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
        ordering = np.arange(p)
        sem = LGSEM(p,1,1,1,1,graph_gen = dag_custom(A, ordering))

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
        noise = np.random.normal([2,0,0,0,0,0], [0,1,1,1,1,1], size=(n,p))
        truth = noise @ M.T
        np.random.seed(42)
        samples = sem.sample(n, do_interventions = np.array([[0,2]]))

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
