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
from src.normal_distribution import NormalDistribution

#---------------------------------------------------------------------
# LGSEM class

class LGSEM:
    """
    Represents an Linear Gaussian SEM. Initialization randomly
    generates a new SEM. sample() generates observational and
    interventional samples from it
    """
    
    def __init__(self, W, variances, intercepts = None, debug=False):
        """Generate a random linear gaussian SEM, given
        - W: weight matrix representing a DAG
        - variances: either a vector of variances or a tuple
          indicating range for uniform sampling
        - intercepts: either a vector of intercepts, a tuple
          indicating the range for uniform sampling or None (zero
          intercepts)
        - graph_gen: the function used to generate the graph, by default
          "generate_dag_avg_dev"
        return a "SEM" object

        """
        self.W = W.copy()
        self.p = len(W)

        # Set variances
        if isinstance(variances, tuple):
            self.variances = np.random.uniform(variances[0], variances[1], size=self.p)
        else:
            self.variances = variances.copy()
            
        # Set intercepts
        if intercepts is None:
            self.intercepts = np.zeros(self.p)
        elif isinstance(intercepts, tuple):
            self.intercepts = np.random.uniform(intercepts[0], intercepts[1], size=self.p)
        else:
            self.intercepts = intercepts.copy()
    
    def sample(self, n=round(1e5), population=False, do_interventions=None, noise_interventions=None, debug=False):
        """
        If population is set to False:
          - Generate n samples from a given Linear Gaussian SEM, under the given
            interventions (by default samples observational data)
        if set to True:
          - Return the "symbolic" joint distribution under the given
            interventions (see class NormalDistribution)
        """
        # Must copy as they can be changed by intervention, but we
        # still want to keep the observational SEM
        W = self.W.copy()
        variances = self.variances.copy()
        intercepts = self.intercepts.copy()

        # Perform do interventions
        if do_interventions is not None:
            targets = do_interventions[:,0].astype(int)
            variances[targets] = 0
            intercepts[targets] = do_interventions[:,1]
            W[:,targets] = 0
            
        # Perform noise interventions
        if noise_interventions is not None:
            targets = noise_interventions[:,0].astype(int)
            intercepts[targets] = noise_interventions[:,1]
            variances[targets] = noise_interventions[:,2]
            W[:,targets] = 0
            
        # Sampling by building the joint distribution
        A = np.linalg.inv(np.eye(self.p) - W.T)
        mean = A @ intercepts
        covariance = A @ np.diag(variances) @ A.T
        distribution = NormalDistribution(mean, covariance)
        if not population:
            return distribution.sample(n)
        else:
            return distribution

#---------------------------------------------------------------------
# DAG Generating Functions

def dag_avg_deg(p, k, w_min, w_max, debug=False, random_state=None, return_ordering=False):
    """
    Generate a random graph with p nodes and average degree k
    """
    np.random.seed(random_state) if random_state is not None else None
    # Generate adjacency matrix as if top. ordering is 1..p
    prob = k / (p-1)
    print("p = %d, k = %0.2f, P = %0.4f" % (p,k,prob)) if debug else None
    A = np.random.uniform(size = (p,p))
    A = (A <= prob).astype(float)
    A = np.triu(A, k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    
    # Permute rows/columns according to random topological ordering
    permutation = np.random.permutation(p)
    # Note the actual topological ordering is the "conjugate" of permutation eg. [3,1,2] -> [2,3,1]
    print("avg degree = %0.2f" % (np.sum(A) * 2 / len(A))) if debug else None
    if return_ordering:
        return (W[permutation, :][:, permutation], np.argsort(permutation))
    else:
        return W[permutation, :][:, permutation]

def dag_full(p, w_min=1, w_max=1, debug=False):
    """Creates a fully connected DAG (ie. upper triangular adj. matrix
    with all ones)"""
    A = np.triu(np.ones((p,p)), k=1)
    weights = np.random.uniform(w_min, w_max, size=A.shape)
    W = A * weights
    return W
