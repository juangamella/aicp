"""
TO CHANGE BEFORE PUBLISHING:
  - color output is not portable, so deactivate it
"""

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
# Unit tests for module icp.py

import unittest
import numpy as np
import copy

from .context import src

# Tested functions
from src.icp import Data, ConfIntervals

class DataTests(unittest.TestCase):

    def setUp(self):
        self.p = 20
        self.N = [2, 3, 4]
        self.n = np.sum(self.N)
        self.target = 3
        environments = []
        for i,ne in enumerate(self.N):
            e = np.tile(np.ones(self.p), (ne, 1))
            e *= (i+1)
            e[:, self.target] *= -1
            environments.append(e)
        self.environments = environments

    def test_basic(self):
        data = Data(self.environments, self.target)
        self.assertEqual(data.n, self.n)
        self.assertTrue((data.N == self.N).all())
        self.assertEqual(data.p, self.p)
        self.assertEqual(data.target, self.target)
        self.assertEqual(data.n_env, len(self.environments))

    def test_memory(self):
        environments = copy.deepcopy(self.environments)
        data = Data(environments, self.target)
        environments[0][0,0] = -100
        data_pooled = data.pooled_data()
        self.assertFalse(data_pooled[0,0] == environments[0][0,0])
        
    def test_targets(self):
        data = Data(self.environments, self.target)
        truth = [-(i+1)*np.ones(ne) for i,ne in enumerate(self.N)]
        truth_pooled = []
        for i,target in enumerate(data.targets):
            self.assertTrue((target == truth[i]).all())
            truth_pooled = np.hstack([truth_pooled, truth[i]])
        self.assertTrue((truth_pooled == data.pooled_targets()).all())
        
    def test_data(self):
        data = Data(self.environments, self.target)
        truth_pooled = []
        for i,ne in enumerate(self.N):
            sample = np.ones(self.p+1)
            sample[:-1] *= (i+1)
            sample[self.target] *= -1
            truth = np.tile(sample, (ne, 1))
            self.assertTrue((truth == data.data[i]).all())
            truth_pooled = truth if i==0 else np.vstack([truth_pooled, truth])
        self.assertTrue((truth_pooled == data.pooled_data()).all())

    def test_split(self):
        data = Data(self.environments, self.target)
        last = 0
        for i,ne in enumerate(self.N):
            (et, ed, rt, rd) = data.split(i)
            # Test that et and ed are correct
            self.assertTrue((et == data.targets[i]).all())
            self.assertTrue((ed == data.data[i]).all())
            # Same as two previous assertions but truth built differently
            truth_et = data.pooled_targets()[last:last+ne]
            truth_ed = data.pooled_data()[last:last+ne, :]
            self.assertTrue((truth_et == et).all())
            self.assertTrue((truth_ed == ed).all())
            # Test that rt and rd are correct
            idx = np.arange(self.n)
            idx = np.logical_or(idx < last, idx >= last+ne)
            truth_rt = data.pooled_targets()[idx]
            truth_rd = data.pooled_data()[idx, :]
            self.assertTrue((truth_rt == rt).all())
            self.assertTrue((truth_rd == rd).all())
            last += ne

class ConfIntervalsTests(unittest.TestCase):
    
    def test_update(self):
        p = 3
        conf_intervals = ConfIntervals(p)
        # Update 1
        conf_intervals.update(set([2]), (np.array([-3, 0]), np.array([3, 0])))
        print(conf_intervals.lower_bound())
        self.assertTrue(nan_equal(conf_intervals.lower_bound(), np.array([np.nan, np.nan, -3, 0])))
        self.assertTrue(nan_equal(conf_intervals.upper_bound(), np.array([np.nan, np.nan, 3, 0])))
        self.assertTrue(nan_equal(conf_intervals.maxmin(), np.array([np.nan, np.nan, -3, 0])))
        self.assertTrue(nan_equal(conf_intervals.minmax(), np.array([np.nan, np.nan, 3, 0])))
        # Update 2
        conf_intervals.update(set([0,1]), (np.array([-1,-1,-1]), np.array([.5, .5, .5])))
        self.assertTrue(nan_equal(conf_intervals.lower_bound(), np.array([-1, -1, -3, -1])))
        self.assertTrue(nan_equal(conf_intervals.upper_bound(), np.array([.5, .5, 3, .5])))
        self.assertTrue(nan_equal(conf_intervals.maxmin(), np.array([-1, -1, -3, 0])))
        self.assertTrue(nan_equal(conf_intervals.minmax(), np.array([.5, .5, 3, 0])))
        # Update 4
        conf_intervals.update(set([0,1,2]), (np.array([-4,-4,-4,-4]), np.array([1,1,1,1])))
        self.assertTrue(nan_equal(conf_intervals.lower_bound(), np.ones(p + 1) * -4))
        self.assertTrue(nan_equal(conf_intervals.upper_bound(), np.array([1, 1, 3, 1])))
        self.assertTrue(nan_equal(conf_intervals.maxmin(), np.array([-1, -1, -3, 0])))
        self.assertTrue(nan_equal(conf_intervals.minmax(), np.array([.5, .5, 1, 0])))

def nan_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True
