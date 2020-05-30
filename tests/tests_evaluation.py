# Copyright 2019 Juan L Gamella

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

from .context import src

# Tested functions
from src.evaluation import intervention_targets

class HelperTests(unittest.TestCase):

    def test_intervention_targets_1(self):
        P = np.random.randint(10, 20, size=50)
        for p in P:
            [response, target] = np.random.choice(p, size=2, replace=False)
            targets = intervention_targets(target, response, p, 0)
            self.assertTrue([target] == targets)

    def test_intervention_targets_2(self):
        P = np.random.randint(10, 20, size=50)
        for p in P:
            max_off_targets = np.random.randint(10)
            [response, target] = np.random.choice(p, size=2, replace=False)
            targets = intervention_targets(target, response, p, max_off_targets)
            self.assertTrue(target in targets)
            self.assertTrue(response not in targets)
            self.assertTrue(len(targets) <= max_off_targets + 1)
            self.assertTrue(len(np.unique(np.array(targets))) == len(targets))
