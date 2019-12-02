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
from src.utils import matrix_block, sampling_matrix

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
