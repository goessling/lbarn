
import numpy as np
from scipy.special import expit
import logitboost_autoregressive_network


# create toy data
n_samples = 1000
np.random.seed(0)
X = np.random.rand(n_samples,3) <= .5
logodds = 5*(2*X[:,0]-1) + 5*(2*X[:,1]-1)
X[:,2] = np.random.rand(n_samples) <= expit(logodds)
X_train = X[:n_samples//3,]
X_valid = X[n_samples//3:2*n_samples//3,]
X_test = X[2*n_samples//3:,]

# train model
model, LLs = logitboost_autoregressive_network.train(X_train, n_leaves=4, n_trees=10, verbose=False)
model = logitboost_autoregressive_network.individual_selection(model, LLs)

# evaluate model
LLs_train = logitboost_autoregressive_network.log_likelihood(model, X_train)
LLs_valid = logitboost_autoregressive_network.log_likelihood(model, X_valid)
LLs_test = logitboost_autoregressive_network.log_likelihood(model, X_test)
print(LLs_train.mean())
print(LLs_valid.mean())
print(LLs_test.mean())

# sample from model
print(logitboost_autoregressive_network.sample(model, 20))


# run via: python3 logitboost_autoregressive_network_test.py
