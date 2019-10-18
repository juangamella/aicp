import numpy as np
import matplotlib.pyplot as plt

WEIGHT_TYPE = np.float64

class LgSem:
    """
    Represents an Linear Gaussian SEM. Initialization randomly
    generates a new SEM. sample() generates observational and
    interventional samples from it
    """
    
    def __init__(self, p, k, w_min, w_max, var_min, var_max, intercepts = None, debug=False):
        """
        Generate a random linear gaussian SEM, given
        - p,k: the number of nodes and avg. degree of the underlying DAG
        - w_min, w_max: lower and upper bounds for sampling weights
        - var_min, var_max: lower and upper bounds for sampling variances
        - intercepts: the "base value" of each variable (0 by default)
        return a "SEM" object
        """
        self.debug = debug
        self.p = p
        
        # Generate DAG
        (A, ordering) = generate_dag_avg_deg(p,k, debug=debug)
        self.ordering = ordering
        
        # Sample weights and update adjacency matrix
        weights = np.random.uniform(w_min, w_max, size=A.shape)
        self.W = A * weights
        
        # Sample variances for noise
        self.variances = np.random.uniform(var_min, var_max, size=len(A))
        
        # Intercepts
        if intercepts is None:
            intercepts = np.zeros(len(A))
            self.intercepts = intercepts
    
    def sample(self, n, do_interventions=None, noise_interventions=None):
        """Generate n samples from a given Linear Gaussian SEM, under the given
        interventions (by default samples observational data)
        """
        debug = self.debug

        # Must copy as they can be changed by intervention, but we
        # still want to keep the observational SEM
        W = self.W.copy()
        variances = self.variances.copy()
        intercepts = self.intercepts.copy()

        # Perform do interventions
        if do_interventions is not None:
            targets = do_interventions[:,0]
            variances[targets] = 0
            intercepts[targets] = do_interventions[:,1]
            W[:,targets] = 0
            
        # Perform noise interventions
        if noise_interventions is not None:
            targets = noise_interventions[:,0]
            intercepts[targets] = noise_interventions[:,1]
            variances[targets] = noise_interventions[:,2]
            W[:,targets] = 0
            
        # Sample
        # First fill X with the noise variables + intercepts
        print("Variances = %s, Intercepts = %s" % (variances, intercepts)) if debug else None
        X = np.random.normal(intercepts, variances, size=(n,p))
        print(X)
        M = W + np.eye(p)
        for i in self.ordering:
            print("Sampling variable %d. Weights %s" % (i,M[:,i])) if debug else None
            X[:,i] = X @ M[:,i]
            
        return X
    
def generate_dag_avg_deg(p, k, debug=False):
    """
    Generate a random graph with p nodes and average degree k
    """
    # Generate adjacency matrix as if top. ordering is 1..p
    prob = k / (p-1)
    print("p = %d, k = %0.2f, P = %0.4f" % (p,k,prob)) if debug else None
    A = np.random.uniform(size = (p,p))
    A = (A <= prob).astype(WEIGHT_TYPE)
    A = np.triu(A, k=1)
    
    # Permute rows/columns according to random topological ordering
    permutation = np.random.permutation(p)
    ordering = np.argsort(permutation)
    # Note the actual topological ordering is the "conjugate" of permutation eg. [3,1,2] -> [2,3,1]
    print("ordering = %s" % ordering) if debug else None
    print("avg degree = %0.2f" % (np.sum(A) * 2 / len(A))) if debug else None
    
    return (A[permutation, :][:, permutation], ordering)
