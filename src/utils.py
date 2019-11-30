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
    idx_rows = np.zeros(n)
    idx_rows[rows] = 1
    idx_cols = np.zeros(m)
    idx_cols[cols] = 1
    mask = np.outer(idx_rows, idx_cols).astype(bool)
    return M[mask].reshape(len(rows), len(cols))

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
# Unit testing

import unittest

class UtilsTests(unittest.TestCase):
    def test_matrix_block(self):
        M = np.array([[11, 12, 13, 14],
                      [21, 22, 23, 24],
                      [31, 32, 33, 34],
                      [41, 42, 43, 44]])
        # Tests
        tests = [(range(4), range(4), M),
                 ([1,2], [3], np.array([[24, 34]]).T),
                 (range(4), [1], M[:,[1]]),
                 ([2], range(4), M[[2],:]),
                 ([0,1], [0,1], np.array([[11, 12], [21, 22]])),
                 ([0,1], [1,3], np.array([[12,14], [22, 24]]))]
        for test in tests:
            (A, B, truth) = test
            #print(A, B, truth, matrix_block(M, A, B))
            self.assertTrue((matrix_block(M, A, B) == truth).all())

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
