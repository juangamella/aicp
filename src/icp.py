import numpy as np

from scipy.stats import ttest_ind as ttest
from scipy.stats import f

from sklearn.linear_model import LinearRegression

from functools import reduce
import itertools
# class ICP():
#     """
#     Contains the implementation of the ICP algorithm.
#     Parameters:

#     Methods:
#       - fit: perform ICP on a given dataset
#     """
    
#     def __init__(debug=False):
#         return None

#     def fit(environments, target, alpha, method='residuals', debug=False):

        
#         if method=='residuals':
            
        

class Data():
    """Class to handle access to the dataset. Takes a list of
    environments (each environment is an np array containing the
    observations) and the index of the target.

    Parameters:
      - p: the number of variables
      - n: the total number of samples
      - N: list with number of samples in each environment
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
        # Split environments into targets and remaining variables (adding a col. of 1s for the intercept)
        self.targets = list(map(lambda e: e[:,target], environments))
        self.data = list(map(lambda e: np.hstack([np.ones((len(e),1)), e[:, np.arange(self.p) != target]]), environments))
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
    stack_fun = np.vstack if axis==0 else np.hstack
    return reduce(lambda acc, array: stack_fun([acc, array]), arrays)
    
def test_hypothesis(s, data, debug=False):
    return np.random.uniform()
#     E = len(environments)
#     p_value_mean = np.zeros(E)
#     p_value_variance = np.zeros_like(E)
#     last = 0
#     for i, e in enumerate(environments):
#         e_data = e
#         others_data = reduce(lambda acc, environments[np.arange(E) != i]
        
#         res_e = pooled_targets(
        
            
from sampling import LGSEM

sem = LGSEM(10,1,1,1,1)
e1 = sem.sample(4)
e2 = sem.sample(3, do_interventions = np.array([[0,2]]))
e3 = sem.sample(2, noise_interventions = np.array([[3,2,1]]))

environments = [e1, e2, e3]
target = 2
debug=True
alpha=0.05
max_predictors = None

###############################################


data = Data(environments, target)
if max_predictors is None:
    max_predictors = data.p

# Find linear coefficients on pooled data
beta = LinearRegression(fit_intercept=False).fit(data.pooled_data(), data.pooled_targets()).coef_

# Test for causal predictor sets of increasing size
base = set(range(data.p))
base.remove(target)
S = base.copy()
max_size = 0
while max_size <= max_predictors and len(S) > 0:
    print("Evaluating candidate sets with length %d" % max_size) if debug else None
    candidates = itertools.permutations(base, max_size)
    for s in candidates:
        p_value = test_hypothesis(s, data, debug=debug)
        holds = p_value <= alpha
        S = S.intersection(s) if holds else S
        print("%s - %s (p=%0.2f) - S = %s" % (s, holds, p_value, S)) if debug else None
        if len(S) == 0:
            break;
    max_size += 1        


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
            sample = np.ones(self.p)
            sample[1:] *= (i+1)
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
            print(idx)
            truth_rt = data.pooled_targets()[idx]
            truth_rd = data.pooled_data()[idx, :]
            self.assertTrue((truth_rt == rt).all())
            self.assertTrue((truth_rd == rd).all())
            last += ne
