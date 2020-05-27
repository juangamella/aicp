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

from sempler import NormalDistribution
import sempler
from src import utils
from src.utils import all_but, sampling_matrix

# Tested functions
from src.population_icp import pooled_regression, markov_blanket, stable_blanket, population_icp

class PopulationICPTests(unittest.TestCase):

    def test_regression_comparison_1(self):
        # Regression over a single environment should yield same
        # results as regressing over that distribution
        graphs = [src.utils.eg1(), src.utils.eg2(), src.utils.eg3(), src.utils.eg4()]
        for g,graph in enumerate(graphs):
            W, _, _ = graph
            p = len(W)
            A = sampling_matrix(W)
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
            W, _, _ = graph
            p = len(W)
            A = sampling_matrix(W)
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
            W, _, mb = graph
            p = len(W)
            A = sampling_matrix(W)
            dist = NormalDistribution(np.arange(p), A@A.T)
            for i in range(p):
                truth = set(mb[i])
                result = set(markov_blanket(i, dist, tol=1e-10))
                # print(truth, result)
                self.assertEqual(truth, result)

    def test_blankets(self):
        np.random.seed(42)
        for p in range(2,8):
            #print("Testing random graph of size %d" %p)
            W = sempler.dag_avg_deg(p,2.5,-1,1)
            sem = sempler.LGANM(W, (0.1,2))
            dist = sem.sample(population=True)
            for i in range(p):
                #print("Testing markov and stable blankets of X_%d" %i)
                (_,_,_,true_mb) = utils.graph_info(i, W)
                # Test markov blanket
                estimated_mb = set(markov_blanket(i, dist, tol=1e-10))
                self.assertEqual(true_mb, estimated_mb)
                # Stable blanket for one env. should be markov blanket
                result = population_icp([dist], i, debug=False, selection='all')
                estimated_sb = stable_blanket(result.accepted, result.mses)
                self.assertEqual(true_mb, estimated_sb)

    def test_blanket_behaviour(self):
        np.random.seed(7)
        for p in range(2,8):
            #print("Testing random graph of size %d" %p)
            W = sempler.dag_avg_deg(p,2.5,-1,1)
            sem = sempler.LGANM(W, (0.1,2))
            dist = sem.sample(population=True)
            for i in range(p):
                #print("Testing markov and stable blankets of X_%d" %i)
                (parents,children,poc,mb) = utils.graph_info(i, W)
                result = population_icp([dist], i, debug=False, selection='all')
                sb_0= stable_blanket(result.accepted, result.mses)
                # Intervening on a parent should leave the stable
                # blanket the same
                if len(parents) > 0:
                    pa = np.random.choice(list(parents))
                    dist_pa = sem.sample(population=True, do_interventions = {pa: (1, 5)})
                    result = population_icp([dist, dist_pa], i, debug=False, selection='all')
                    sb_pa = stable_blanket(result.accepted, result.mses)
                    self.assertEqual(sb_0, sb_pa)
                # Intervening on a parent of a child (that is not a child) should leave the stable
                # blanket the same
                only_poc = poc.difference(children)
                if len(only_poc) > 0:
                    pc = np.random.choice(list(only_poc))
                    dist_pc = sem.sample(population=True, do_interventions = {pc: (1, 5)})
                    result = population_icp([dist, dist_pc], i, debug=False, selection='all')
                    sb_pc = stable_blanket(result.accepted, result.mses)
                    self.assertEqual(sb_0, sb_pc)
                # Intervening on a child should affect the stable blanket
                if len(children) > 0:
                    ch = np.random.choice(list(children))
                    dist_ch = sem.sample(population=True, do_interventions = {ch: (1, 5)})
                    result = population_icp([dist, dist_ch], i, debug=False, selection='all')
                    sb_ch= stable_blanket(result.accepted, result.mses)
                    _, descendants, _, _ = utils.graph_info(ch, W)
                    for d in descendants.union({ch}):
                        self.assertTrue(d not in sb_ch)
