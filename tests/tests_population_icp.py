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
# Unit testing for module population_icp.py

import unittest
import numpy as np

from .context import src

from src.normal_distribution import NormalDistribution
from src.utils import all_but, sampling_matrix

# Tested functions
from src.population_icp import pooled_regression, markov_blanket

class PopulationICPTests(unittest.TestCase):

    def test_regression_comparison_1(self):
        # Regression over a single environment should yield same
        # results as regressing over that distribution
        graphs = [src.utils.eg1(), src.utils.eg2(), src.utils.eg3(), src.utils.eg4()]
        for g,graph in enumerate(graphs):
            W, ordering, _, _ = graph
            p = len(W)
            A = sampling_matrix(W, ordering)
            dist = NormalDistribution(np.arange(p), A@A.T)
            envs = [dist]
            for i in range(p):
                for l in range(p+1):
                    S = [k for k in range(l) if k != i]
                    #print("eg%d: y=%d, S=%s" % (g+1,i,S))
                    (true_coefs, true_intercept) = dist.regress(i, S)
                    true_mse = dist.mse(i, S)
                    (pooled_coefs, pooled_intercept, pooled_mse) = pooled_regression(i,S,envs)
                    self.assertTrue(np.allclose(true_coefs, pooled_coefs))
                    self.assertTrue(np.allclose(true_intercept, pooled_intercept))
                    self.assertTrue(np.allclose(true_mse, pooled_mse))

    def test_regression_comparison_2(self):
        # Regression over copies of the same environment should yield
        # same results as regressing over that distribution
        graphs = [src.utils.eg1(), src.utils.eg2(), src.utils.eg3(), src.utils.eg4()]
        for g,graph in enumerate(graphs):
            W, ordering, _, _ = graph
            p = len(W)
            A = sampling_matrix(W, ordering)
            dist = NormalDistribution(np.arange(p), A@A.T)
            envs = [dist, dist, dist]
            for i in range(p):
                for l in range(p+1):
                    S = [k for k in range(l) if k != i]
                    #print("eg%d: y=%d, S=%s" % (g+1,i,S))
                    (true_coefs, true_intercept) = dist.regress(i, S)
                    true_mse = dist.mse(i, S)
                    (pooled_coefs, pooled_intercept, pooled_mse) = pooled_regression(i,S,envs)
                    self.assertTrue(np.allclose(true_coefs, pooled_coefs))
                    self.assertTrue(np.allclose(true_intercept, pooled_intercept))
                    self.assertTrue(np.allclose(true_mse, pooled_mse))

    def test_markov_blanket(self):
        graphs = [src.utils.eg1(), src.utils.eg2(), src.utils.eg3(), src.utils.eg4()]
        for g,graph in enumerate(graphs):
            W, ordering, _, mb = graph
            p = len(W)
            A = sampling_matrix(W, ordering)
            dist = NormalDistribution(np.arange(p), A@A.T)
            for i in range(p):
                truth = set(mb[i])
                result = set(markov_blanket(i, dist, tol=1e-10))
                # print(truth, result)
                self.assertEqual(truth, result)
