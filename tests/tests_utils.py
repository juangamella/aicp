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

#os.chdir('/home/juan/ETH/code_semester_project/src')

#---------------------------------------------------------------------
# Unit testing for module utils

import unittest
import numpy as np

from .context import src
from src.sampling import dag_avg_deg

# Tested functions
from src import utils
from src.utils import matrix_block, sampling_matrix, nonzero, all_but, graph_info, stable_blanket

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
                 ([0,1], [1,3], np.array([[12,14], [22, 24]])),
                 # Test order of indices is also respected
                 (range(3,-1,-1), range(3,-1,-1), M[::-1,::-1]),
                 (range(3,-1,-1), range(4), M[::-1,:]),
                 (range(3,-1,-1), [1], M[::-1,[1]]),
                 ([2], range(3,-1,-1), M[[2],::-1]),
                 ([1,0], [0,1], np.array([[21, 22], [11, 12]])),
                 ([0,1], [3,1], np.array([[14,12], [24, 22]]))]
        for test in tests:
            (A, B, truth) = test
            #print(A, B, truth, matrix_block(M, A, B))
            self.assertTrue((matrix_block(M, A, B) == truth).all())

    def test_sampling_matrix(self):
        W = np.array([[0,0,1,0],
                      [0,0,1,0],
                      [0,0,0,1],
                      [0,0,0,0]])
        truth = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [1,1,1,0],
                          [1,1,1,1]])
        self.assertTrue((truth == sampling_matrix(W, range(4))).all())

    def test_nonzero(self):
        tol = 1e-12
        A = np.array([0, 1, tol, -1, tol/2, -tol*2])
        self.assertTrue((np.array([1,3,5]) == nonzero(A)).all())

    def test_all_but(self):
        self.assertTrue([0, 1, 3, 4] == all_but(2,5))
        self.assertTrue([0, 3, 4] == all_but([1,2],5))

    def test_graph_info_1(self):
        W = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 1., 0., 1., 0., 0.],
                      [0., 1., 0., 0., 0., 1., 1., 0.],
                      [1., 0., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 0., 0., 0., 1., 1., 0., 0.]])
        ordering = np.array([7, 4, 2, 3, 6, 5, 0, 1])
        true_parents = [{4,6,7},
                        {0,2,3},
                        set(),
                        {2},
                        {7},
                        {2,3,7},
                        {3,4},
                        set()]
        true_children = [{1},
                         set(),
                         {1,3,5},
                         {1,5,6},
                         {0,6},
                         set(),
                         {0},
                         {0,4,5}]
        true_poc = [{2,3},
                    set(),
                    {0,3,7},
                    {0,2,4,7},
                    {3,6,7},
                    set(),
                    {4,7},
                    {2,3,4,6}]
        true_mb = [{4,6,7,1,2,3},
                   {0,2,3},
                   {1,3,5,0,7},
                   {0,1,2,4,5,6,7},
                   {7,0,6,7,3},
                   {2,3,7},
                   {3,4,0,7},
                   {0,4,5,2,3,6}]
        for i in range(len(W)):
            #print("Testing info for node %d" %i)
            (parents, children, poc, mb) = graph_info(i, W)
            #print(parents, children, poc, mb)
            self.assertEqual(parents, true_parents[i])
            self.assertEqual(children, true_children[i])
            self.assertEqual(poc, true_poc[i])
            self.assertEqual(mb, true_mb[i])

    def test_graph_info_2(self):
        graphs = [utils.eg1(), utils.eg2(), utils.eg3(), utils.eg4(), utils.eg5(), utils.eg6()]
        for k,graph in enumerate(graphs):
            (W, ordering, true_parents, true_mb) = graph
            for i in range(len(W)):
                #print("%d Testing info for node %d" % (k+1,i))
                (parents, children, poc, mb) = graph_info(i, W)
                #print(parents, children, poc, mb)
                self.assertEqual(parents, set(true_parents[i]))
                self.assertEqual(mb, set(true_mb[i]))

    def test_stable_blanket(self):
        W, _, _, markov_blanket = utils.eg5()
        target = 3
        intervened_variables = [set(), {2}, {7}, {6}, {5}]
        truth = [markov_blanket,
                 markov_blanket,
                 markov_blanket,
                 {2, 5, 7},
                 {2}]
        interventions = set()
        for i, var in enumerate(intervened_variables):
            interventions.update(var)
            self.assertTrue(truth[i], stable_blanket(target, W, interventions))
