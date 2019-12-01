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

os.chdir('/home/juan/ETH/code_semester_project/src')

#---------------------------------------------------------------------
# Unit testing for module normal_distribution.py

import unittest
import numpy as np
from utils import sampling_matrix

from normal_distribution import NormalDistribution

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

    def run_mse_tests(self, joint, tests):
        for test in tests:
            (y, Xs, true_mse) = test
            result = joint.mse(y, Xs)
            if not np.allclose(true_mse, result):
                print(test)
                self.fail("%0.16f != %0.16f" % (true_mse, result))

    def mse_properties_test(self, W, parents, markov_blankets, sample_weights=True):
        # Given a weight matrix, parents of each node and markov
        # blankets of each node, test basic properties of the MSE
        # PRE: It is assumed that [1,...,p] is the causal ordering
        p = len(W)
        ordering = np.arange(p)
        np.random.seed(42)
        if sample_weights:
            W = W * (np.random.uniform(size=W.shape) + 1) # avoid 0 weights (causal minimality)
        A = sampling_matrix(W, ordering)
        covariance = A @ A.T
        mean = np.random.uniform(size=p)
        joint = NormalDistribution(mean, covariance)
        all_but = lambda k: [i for i in range(p) if i != k]
        # Tests
        tests = []
        # When no regressors, the MSE is just the marginal variance
        tests += [(i, [], covariance[i,i]) for i in range(p)]
        # Regressing a variable on itself should give 0 MSE
        tests += [(i, [i], 0) for i in range(p)]
        # Regressing on all variables (including itself) should also
        # give 0 MSE
        tests += [(i, range(p), 0) for i in range (p)]
        # Regressing on the parents should yield MSE equal to
        # the variance of the noise variable (1 in this case)
        tests += [(i, pa, 1) for i,pa in enumerate(parents)]
        # Regressing on additional variables to the Markov Blanket
        # should give the same result
        tests += [(i, all_but(i), joint.mse(i, mb)) for i,mb in enumerate(markov_blankets)]
        self.run_mse_tests(joint, tests)
        # Changing the mean of the noise variables should not change
        # the MSE
        joint.mean = np.random.uniform(size=p)
        self.run_mse_tests(joint, tests)

    def regression_properties_test(self, W, parents, markov_blankets):
        # Test basic properties of the regression coefficients.
        # PRE: Assumed [1..p] is a valid causal ordering
        p = len(W)
        np.random.seed(42)
        W = W * (np.random.uniform(size=W.shape) + 1) # avoid 0 weights (causal minimality)
        A = sampling_matrix(W, np.arange(p))
        covariance = A @ A.T
        mean = np.random.uniform(size=p)
        joint = NormalDistribution(mean, covariance)
        all_but = lambda k: [i for i in range(p) if i != k]
        # Tests
        # Regressing on the parents should return their weights as coefficients
        for i,pa in enumerate(parents):
            (coefs, intercept) = joint.regress(i, pa)
            if not np.allclose(W[pa, i], coefs[pa]):
                self.fail("%s ~ %s. Expected: %s Got: %s" % (i, pa, W[pa, i], coefs[pa]))
        # Regressing on all variables should yield weight 1 on the
        # target and 0 intercept
        for i in range(p):
            (coefs, intercept) = joint.regress(i, range(p))
            truth = np.zeros(p)
            truth[i] = 1
            if not np.allclose(truth, coefs):
                self.fail("%s ~ %s. Expected: %s Got: %s" % (i, pa, truth, coefs))
            if not np.allclose(0, intercept):
                self.fail("%s ~ %s. Expected: %s Got: %s" % (i, pa, 0, intercept))
        # Regressing on all variables except target should yield Markov blanket
        for i,mb in enumerate(markov_blankets):
            mb.sort()
            (coefs, intercept) = joint.regress(i, all_but(i))
            non_mb = [i for i in range(p) if i not in mb]
            # Check vars. outside MB have 0 coefficient
            truth = np.zeros(len(non_mb))
            if not np.allclose(truth, coefs[non_mb]):
                self.fail("%s ~ %s. Expected: %s Got: %s" % (i, mb, truth, coefs[non_mb]))
            # Check that vars. inside MB have non-zero coefficient (we
            # assumed causal minimality above)
            tol = 1e-10
            if not (np.abs(coefs[mb]) > tol).all():
                self.fail("%s ~ %s. Expected all non-zero. Got: %s" % (i, mb, coefs[mb]))

    def test_mse_1(self):
        # Test a set of regressions
        covariance = np.array([[1, 0, 1, 1],
                               [0, 1, 1, 1],
                               [1, 1, 3, 3],
                               [1, 1, 3, 4]])
        mean = np.array([0,0,0,0])
        joint = NormalDistribution(mean, covariance)
        tests = [
            # Regressing X0
            (0, [1], 1), # indep. var
            (0, [2], 2/3), # children
            (0, [1,2], 0.5), # markov blanket
            (0, [2,1], 0.5), # markov blanket (test order does not matter)
            # Regressing X1
            (1, [0], 1), # indep. var
            (1, [2], 2/3), # children
            (1, [0,2], 0.5), # markov blanket
            (1, [2,0], 0.5), # markov blanket (test order does not matter)
            # Regressing X2
            (2, [0], 2), # parent
            (2, [1], 2), # parent
            (2, [0,1], 1), # parents
            (2, [0,1,3], 0.5), # markov blanket
            (2, [3], 0.75), # child
            # Regressing X3
            (3, [0], 3), # X0
            (3, [0], 3), # X1
            (3, [0,1], 2), # X0 and X1
            (3, [1,0], 2), # X1 and X0 (test order does not matter)
            (3, [2,1], 1), # markov blanket + d-separated
            (3, [1,2], 1), # markov blanket + d-separated (test order does not matter)
            (3, [2,0], 1), # markov blanket + d-separated
            (3, [2], 1)] # markov blanket
        for test in tests:
            (y, Xs, true_mse) = test
            self.assertEqual(true_mse, joint.mse(y, Xs))
        # Changing the mean of the noise variables should not change
        # the results
        joint.mean = np.array([1,2,3,4])
        for test in tests:
            (y, Xs, true_mse) = test
            self.assertEqual(true_mse, joint.mse(y, Xs))
            
    def test_eg1(self):
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
        self.mse_properties_test(W, parents, markov_blankets)
        self.mse_properties_test(W, parents, markov_blankets, sample_weights=False)
        self.regression_properties_test(W, parents, markov_blankets)
        
    def test_eg2(self):
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
        self.mse_properties_test(W, parents, markov_blankets)
        self.mse_properties_test(W, parents, markov_blankets, sample_weights=False)
        self.regression_properties_test(W, parents, markov_blankets)

    def test_eg3(self):
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
        self.mse_properties_test(W, parents, markov_blankets)
        self.mse_properties_test(W, parents, markov_blankets, sample_weights=False)
        self.regression_properties_test(W, parents, markov_blankets)

    def test_eg4(self):
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
        self.mse_properties_test(W, parents, markov_blankets)
        self.mse_properties_test(W, parents, markov_blankets, sample_weights=False)
        self.regression_properties_test(W, parents, markov_blankets)
