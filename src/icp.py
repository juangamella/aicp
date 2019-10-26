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

from scipy.stats import ttest_ind as ttest
from scipy.stats import f

from sklearn.linear_model import LinearRegression

from functools import reduce
import itertools

#---------------------------------------------------------------------
# "Public" API: icp function

def icp(environments, target, alpha=0.05, max_predictors=None, debug=False):
    """
    ICP on the given target using data from the given environments
    """
    data = Data(environments, target)
    # Test for causal predictor sets of increasing size
    max_predictors = data.p-1 if max_predictors is None else max_predictors
    base = set(range(data.p))
    base.remove(target)
    S = base.copy()
    max_size = 0
    while max_size <= max_predictors and len(S) > 0:
        print("Evaluating candidate sets with length %d" % max_size) if debug else None
        candidates = itertools.combinations(base, max_size)
        for s in candidates:
            # Find linear coefficients on pooled data
            beta = regress(s, data)
            assert((beta[list(base.difference(s))] == 0).all())
            p_value = test_hypothesis(beta, data, debug=debug)
            rejected = p_value < alpha
            S = S.intersection(s) if not rejected else S
            beta_str = np.array_str(beta, precision=2)
            print("%s - %s (p=%0.2f) - S = %s %s" % (s, not rejected, p_value, S, beta_str)) if debug else None
            #if len(S) == 0:
            #        break;
        max_size += 1
    return S

# Support functions to icp

def test_hypothesis(beta, data, debug=False):
    """Test hypothesis for a vector of coefficients beta, using the t-test for the mean
    and f-test for the variances, and returning the p-value

    """
    mean_pvalues = np.zeros(data.n_env)
    var_pvalues = np.zeros(data.n_env)
    for i in range(data.n_env):
        (env_targets, env_data, others_targets, others_data) = data.split(i)
        residuals_env = env_targets - env_data @ beta
        residuals_others = others_targets - others_data @ beta
        mean_pvalues[i] = t_test(residuals_env, residuals_others)
        var_pvalues[i] = f_test(residuals_env, residuals_others)
        assert(mean_pvalues[i] <= 1)
        assert(var_pvalues[i] <= 1)
    # Combine via bonferroni correction
    pvalue_mean = np.mean(mean_pvalues)
    pvalue_var = np.mean(var_pvalues)
    # Return two times the smallest p-value
    return min(pvalue_mean, pvalue_var) * 2

def regress(s, data, pooling=True, debug=False):
    """
    Perform linear regression of data.target over the variables indexed by s
    """
    supp = list(s) + [data.p]
    if pooling:
        X = data.pooled_data()[:,supp]
        Y = data.pooled_targets()
    beta = np.zeros(data.p+1)
    beta[supp] = LinearRegression(fit_intercept=False).fit(X, Y).coef_
    return beta

def t_test(X,Y):
    """Return the test statistic and p-value of the two sample f-test for
    the given sample"""
    result = ttest(X, Y, equal_var=False)
    return result.pvalue

def f_test(X,Y):
    """Return the test statistic and p-value of the two sample g-test for
    the given sample"""
    X = X[np.isfinite(X)]
    Y = Y[np.isfinite(Y)]
    F = np.var(X, ddof=1) / np.var(Y, ddof=1)
    p = f.cdf(F, len(X)-1, len(Y)-1)
    return  2*min(p, 1-p)

#---------------------------------------------------------------------
# Data class and its support functions

class Data():
    """Class to handle access to the dataset. Takes a list of
    environments (each environment is an np array containing the
    observations) and the index of the target.

    Parameters:
      - p: the number of variables
      - n: the total number of samples
      - N: list with number of samples in each environment
      - n_env: the number of environments
      - targets: list with the observations of the target in each environment
      - data: list with the observations of the other vars. in each environment
      - target: the index of the target variable

    """
    def __init__(self, environments, target):
        """Initializes the object by separating the observations of the target
        from the rest of the data, and obtaining the number of
        variables, number of samples per environment and total number
        of samples.

        Arguments:
          - environments: list of np.arrays of dim. (n_e, p), each one
            containing the data of an environment. n_e is the number of
            samples for that environment and p is the number of variables.
          - target: the index of the target variable
        """
        self.N = np.array(list(map(len, environments)))
        self.p = environments[0].shape[1]
        self.n = np.sum(self.N)
        self.n_env = len(environments)
        # Extract targets and add a col. of 1s for the intercept
        self.targets = list(map(lambda e: e[:,target], environments))
        self.data = list(map(lambda e: np.hstack([e, np.ones((len(e),1))]), environments))
        self.target = target

    def pooled_data(self):
        """Returns the observations of all variables except the target,
        pooled."""
        return pool(self.data, 0)

    def pooled_targets(self):
        """Returns all the observations of the target variable, pooled."""
        return pool(self.targets, 1)

    def split(self, i):
        """Splits the dataset into targets/data of environment i and
        targets/data of other environments pooled together."""
        rest_data = [d for k,d in enumerate(self.data) if k!=i]
        rest_targets = [t for k,t in enumerate(self.targets) if k!=i]
        self.data[i]
        return (self.targets[i], self.data[i], pool(rest_targets, 1), pool(rest_data, 0))

def pool(arrays, axis):
    """Takes a list() of numpy arrays and returns them in an new
    array, stacked along the given axis.
    """
    if len(arrays) == 1:
        return arrays[0]
    else:
        stack_fun = np.vstack if axis==0 else np.hstack
        return reduce(lambda acc, array: stack_fun([acc, array]), arrays)

#---------------------------------------------------------------------
# Unit testing

import unittest
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
