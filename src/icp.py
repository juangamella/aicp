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

import numpy as np

from scipy.stats import ttest_ind as ttest
from scipy.stats import f
from scipy.stats import t

from sklearn.linear_model import LinearRegression

from functools import reduce
import itertools

import copy

from termcolor import colored
#---------------------------------------------------------------------
# "Public" API: icp function

def icp(environments, target, alpha=0.01, max_predictors=None, debug=False, stop_early=True):
    """
    ICP on the given target using data from the given environments
    """
    data = Data(environments, target)
    # Test for causal predictor sets of increasing size
    max_predictors = data.p-1 if max_predictors is None else max_predictors
    base = set(range(data.p))
    base.remove(target)
    S = base.copy()
    accepted = [] # To store the accepted sets
    rejected = [] # To store the sets that were rejected
    mses = [] # To store the MSE of the accepted sets
    max_size = 0
    conf_intervals = ConfIntervals(data.p)
    while max_size <= max_predictors and (len(S) > 0 or not stop_early):
        print("Evaluating candidate sets with length %d" % max_size) if debug else None
        candidates = itertools.combinations(base, max_size)
        for s in candidates:
            s = set(s)
            # Find linear coefficients on pooled data
            (beta, error) = regress(s, data)
            assert((beta[list(base.difference(s))] == 0).all())
            p_value = test_hypothesis(beta, data, debug=debug)
            reject = p_value < alpha
            if reject:
                rejected.append(s)
            else:
                accepted.append(s)
                S = S.intersection(s)
                mses.append(error)
                if max_size !=0:
                    intervals = confidence_intervals(s, beta, data, alpha)
                    print(intervals)
                    conf_intervals.update(s, intervals)
            if debug:
                color = "red" if reject else "green"
                beta_str = np.array_str(beta, precision=2)
                set_str = "rejected" if reject else "accepted"
                msg = colored("%s %s" % (s, set_str), color) + " - (p=%0.2f) - S = %s %s MSE: %0.4f" % (p_value, S, beta_str, error)
                print(msg)
            if len(S) == 0 and stop_early:
                break;
        max_size += 1
    return Result(S, accepted, rejected, mses, conf_intervals)

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
    pvalue_mean = min(mean_pvalues) * data.n_env
    pvalue_var = min(var_pvalues) * data.n_env
    # Return two times the smallest p-value
    return min(pvalue_mean, pvalue_var) * 2

def regress(s, data, pooling=True, debug=False):
    """
    Perform the linear regression of data.target over the variables indexed by s
    """
    supp = list(s) + [data.p] # support is pred. set + intercept
    if pooling:
        X = data.pooled_data()[:,supp]
        Y = data.pooled_targets()
    beta = np.zeros(data.p+1)
    beta[supp] = LinearRegression(fit_intercept=False).fit(X, Y).coef_
    error = mse(Y, data.pooled_data() @ beta)
    return beta, error

def mse(true, pred):
    return np.sum((true - pred)**2) / len(true)

def t_test(X,Y):
    """Return the p-value of the two sample f-test for
    the given sample"""
    result = ttest(X, Y, equal_var=False)
    return result.pvalue

def f_test(X,Y):
    """Return the p-value of the two sample g-test for
    the given sample"""
    X = X[np.isfinite(X)]
    Y = Y[np.isfinite(Y)]
    F = np.var(X, ddof=1) / np.var(Y, ddof=1)
    p = f.cdf(F, len(X)-1, len(Y)-1)
    return  2*min(p, 1-p)

def confidence_intervals(s, beta, data, alpha):
    """Compute the confidence intervals of the regression coefficients
    (beta) of a predictor set s, given the level alpha.

    Under Gaussian errors, the confidence intervals are given by
    beta +/- delta, where

    delta = quantile * variance of residuals @ diag(inv. corr. matrix)

    and variance and corr. matrix of residuals are estimates
    """
    s = list(s)
    supp = s + [data.p] # Support is pred. set + intercept
    beta = beta[supp]
    # Quantile term
    dof = data.n - len(s) - 1
    quantile = t.ppf(1-alpha/2/len(s), dof)
    # Residual variance term
    Xs = data.pooled_data()[:,supp]
    residuals = data.pooled_targets() - Xs @ beta
    variance = np.var(residuals)
    # Corr. matrix term
    sigma = np.diag(np.linalg.inv(Xs.T @ Xs))
    # Compute interval
    delta = quantile * np.sqrt(variance) * sigma
    return (beta - delta, beta + delta)
    
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
        environments = copy.deepcopy(environments) # ensure the stored data is immutable
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
# Confidence intervals class
class ConfIntervals():
    """Class to keep and update confidence intervals for the regression
    coefficients of every variable
    
    Parameters:
    - 
    """

    def __init__(self, p):
        """Initializes the arrays used to store the lower and upper bounds"""
        self.p = p 
        self.lwr = []
        self.upr = []

    def update(self, s, bounds):
        """Given a set of variables s and new bounds, update the stored bounds
        by taking the union. Note that "bounds" also include the bounds for
        the intercept, although it is not specified in s
        """
        (lwr, upr) = bounds
        supp = list(s) + [self.p] # support is predictor set s + intercept
        lwr_new = np.ones(self.p+1)*np.nan
        upr_new = np.ones(self.p+1)*np.nan
        lwr_new[supp] = lwr
        upr_new[supp] = upr
        self.lwr.append(lwr_new)
        self.upr.append(upr_new)
        return (self.lwr, self.upr)

    def lower_bound(self):
        lower_bounds = np.array(self.lwr)
        return np.nanmin(lower_bounds, axis=0)

    def upper_bound(self):
        upper_bounds = np.array(self.upr)
        return np.nanmax(upper_bounds, axis=0)

    def maxmin(self):
        lower_bounds = np.array(self.lwr)
        return np.nanmax(lower_bounds, axis=0)

    def minmax(self):
        upper_bounds = np.array(self.upr)
        return np.nanmin(upper_bounds, axis=0)

#---------------------------------------------------------------------
# Results class

class Result():
    """Class to hold the estimate produced by ICP and any additional information"""
    def __init__(self, estimate, accepted, rejected, mses, conf_intervals=None):
        self.estimate = estimate # The estimate produced by ICP ie. intersection of accepted sets
        self.accepted = accepted # Accepted sets
        self.rejected = rejected # Rejected sets
        self.mses = np.array(mses) # MSE of the accepted sets
        self.conf_intervals = conf_intervals # Confidence intervals
