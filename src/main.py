import networkx as nx
import matplotlib.pyplot as plt
from src.sampling import LGSEM
from src.icp import icp
import numpy as np



W = np.array([[0, 1, -1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])

# W = np.array([[0, 1, -1, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 1, 0]])

p = len(W)
ordering = np.arange(p)
sem = LGSEM(W,ordering,(1,1))
n = round(1e4)
e = sem.sample(n)
e0 = sem.sample(n, noise_interventions = np.array([[0,10,1]]))
e1 = sem.sample(n, noise_interventions = np.array([[1,10,1]]))
e2 = sem.sample(n, do_interventions = np.array([[2,10,1]]))
e4 = sem.sample(n, do_interventions = np.array([[4,10,1]]))
#e5 = sem.sample(n, do_interventions = np.array([[5,10,1]]))
print(sem.W)
#_ = plot_graph(sem)

environments = [e, e1, e0]#, e2, e3, e4
target = 3

(S, accepted, mses, conf_intervals) = icp(environments, target, alpha=0.01, max_predictors=None, debug=True, stop_early=False)
print("Intersection: %s" % S)

def ratios(p, accepted):
    one_hot = np.zeros((len(accepted), p))
    for i,s in enumerate(accepted):
        one_hot[i, s] = 1
    return one_hot.sum(axis=0) / len(accepted)
        
R = ratios(p, accepted)
for i,r in enumerate(R):
    print("X%d = %d/%d=%0.2f" % (i, r*p, p, r))

mses = np.array(mses)
for i in np.argsort(mses):
    print("%s %0.5f" % (accepted[i], mses[i]))


# class RandomPolicy():
#     def __init__(self, p, target):
#         self.p = p
#         self.target = target
#         self.vars = [i for i in range(p) if i != target]

#     def first(self, obs_data):
#         return np.random.choice(self.vars)
    
#     def next(self, accepted, intervened):
#         not_intervened = [i for i in self.vars if i not in intervened]
#         if not not_intervened: # ie. not_intervened is empty
#             return None
#         else:
#             return np.random.choice(not_intervened)

# def test_policy(policy, sem, target, alpha = 0.01, n=round(1e5), n_i = round(1e5), max_iter = 100):
#     obs_data = sem.sample(n) # sample observational data
#     next_intervention = policy.first(obs_data) # get first intervention
#     i = 0
#     done = False
#     environments = [obs_data]
#     intervened = []
#     Accepted = []
#     while (i < max_iter) and next_intervention is not None:
#         print("\nIntervening on %d\n" % next_intervention)
#         e = sem.sample(n_i, do_interventions= np.array([[next_intervention, 10]]))
#         environments.append(e)
#         (S, accepted, conf_intervals) = icp(environments, target, alpha, max_predictors=None, debug=False, stop_early=False)
#         intervened.append(next_intervention)
#         next_intervention = policy.next(accepted, intervened)
#         Accepted.append(accepted)
#     return intervened, Accepted
    
    
# #test_policy(RandomPolicy(p, target), sem, target, max_iter=6)


class PolicyEvaluationResults():
    """Class to contain all information resulting from evaluating a policy
    over a set of cases"""

    def __init__(self, policy, cases):
        self.policy = policy
        self.cases = cases
        # Info
        self.estimates = [] # estimate produced by the policy
        self.interventions = [] # interventions used by the policy
        self.scores = None # score of the estimate (using metric)
        self.times = [] # time used by the policy
        
    def compute_scores(metric=None):
        """Compute (and store) the scores of the estimates wrt. to a given
        metric
        """
        if metric is None:
            metric = jaccard_distance
        scores = []
        for i,case in enumerate(self.cases):
            scores.append(metric(case.truth, estimates[i]))
        self.scores = scores.copy()
        return scores
        
class TestCase():
    """Object that represents a test case
    ie. SEM + target + expected result
    """
    def __init(self, sem, target, truth):
        self.sem = sem
        self.target = target
        self.truth = truth

def jaccard_distance(A, B):
    """Compute the jaccard distance between sets A and B"""
    return len(set.intersection(A,B)) / len(set.union(A,B))

import time
def evaluate_policy(policy, cases, debug=False):
    """Evaluate a policy over the given test cases, returning a
    PolicyEvaluationResults object containing the results
    """
    results = PolicyEvaluationResults(policy, cases)
    for i,case in enumerate(cases):
        print("%0.2f Evaluating policy on test case %d..." % i/len(cases)*100, end="") if debug else None
        start = time.time()
        (estimate, interventions) = run_policy(policy, case.sem, case.target)
        end = time.time()
        elapsed = end - start
        print(" done (%0.2f seconds)" % elapsed) if debug else None
        results.estimates.append(estimate)
        results.interventions.append(interventions)
        results.times.append(times)
    return results

