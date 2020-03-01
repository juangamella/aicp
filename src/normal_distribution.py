# Copyright 2020 Juan Luis Gamella Martin

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
from src.utils import matrix_block

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
        # Parse params
        X = np.atleast_1d(X)
        # Compute marginal mean/variance
        mean = self.mean[X].copy()
        covariance = matrix_block(self.covariance, X, X).copy()
        return NormalDistribution(mean, covariance)

    def conditional(self, Y, X, x):
        """Return the conditional distribution of the variables with indices Y
        given observations x of the variables with indices X

        """
        # Parse params
        Y = np.atleast_1d(Y)
        X = np.atleast_1d(X)
        x = np.atleast_1d(x)
        if len(X) == 0:
            return self.marginal(Y)
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

    def regress(self, y, Xs):
        """Compute the coefficients and intercept of regressing y on
        predictors Xs, where the joint distribution is a multivariate
        Gaussian
        """
        coefs = np.zeros(self.p)
        # If predictors are given, perform regression, otherwise just fit
        # intercept
        if Xs:
            cov_y_xs = matrix_block(self.covariance, [y], Xs)
            cov_xs = matrix_block(self.covariance, Xs, Xs)
            coefs[Xs] = cov_y_xs @ np.linalg.inv(cov_xs)
        intercept = self.mean[y] - coefs @ self.mean
        return (coefs, intercept)

    def mse(self, y, Xs):
        """Compute the population MSE of regressing y on predictors Xs, where
        the joint distribution is a multivariate Gaussian
        """
        var_y = self.covariance[y,y]
        # Compute regression coefficients when regressing on Xs
        (coefs_xs, _) = self.regress(y, Xs)
        # Covariance matrix
        cov = self.covariance
        # Computing the MSE
        mse = var_y + coefs_xs @ cov @ coefs_xs.T - 2 * cov[y,:] @ coefs_xs.T
        return mse

    def equal(self, dist, tol=1e-7):
        return np.allclose(self.mean, dist.mean) and np.allclose(self.covariance, dist.covariance)
