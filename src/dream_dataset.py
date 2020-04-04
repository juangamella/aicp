import pickle
import numpy as np
from src.icp import icp
from src import policy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
(e, ei) = pickle.load(open("scratch/dream4_size10_3.pickle", "rb"))

# Explore
x = 1
y = 4

# Plot datapoints
plt.subplot(211)
plt.scatter(e[:,x], e[:,y], marker='x', label="Observational")
pooled = [e]
for i, env in enumerate(ei):
    plt.scatter(env[:,x], env[:,y], label="int. on G%d" % (i+1))
    pooled.append(env) if i != y else None
plt.xlabel("G%d" % (x+1))
plt.ylabel("G%d" % (y+1))
plt.legend()
plt.title("Datapoints G%d vs. G%d (in graph G%d is NOT parent of G%d)" % (x+1,y+1,x+1,y+1))
    
# Plot residuals
pooled = np.vstack(pooled)
reg = LinearRegression().fit(pooled[:,[x]], pooled[:,y])
beta = reg.coef_
mu = reg.intercept_
plt.plot([pooled.min(), pooled.max()], [pooled.min(), pooled.max()]*beta + mu, color='red')

plt.subplot(212)
plt.scatter(e[:,x], e[:,y] - beta * e[:,x] - mu, marker='x', label="Observational")
for i, env in enumerate(ei):
    plt.scatter(env[:,x], env[:,y] - beta * env[:,x] - mu, label="int. on G%d" % (i+1))
plt.legend()
plt.plot([pooled.min(), pooled.max()], [0,0], color='red')
plt.xlabel("G%d" % (x+1))
plt.ylabel("Residuals")
plt.title("Residuals G%d ~ G%d" % (y+1,x+1))

plt.show(block=False)


# Experiment parameters
target = 4
alpha = 0.01
p = 10

result = icp([e, ei[2]], target, max_predictors = 1, alpha=alpha, selection = 'all', debug=True)

R = policy.ratios(p, result.accepted)
print(R)
for i,r in enumerate(R):
    print("X%d = %d/%d=%0.4f" % (i, r*len(result.accepted), len(result.accepted), r))
result.estimate


