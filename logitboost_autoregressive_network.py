
#
# Logitboost autoregressive network.
#


import heapq

import numpy as np

import logitboost



def train(X, n_leaves, n_trees, step_size=1, use_gb=False, verbose=True):
    """
    Trains a logitboost autoregressive network.

    Parameters
    ----------
    X : (n_samples,n_dims) boolean array
        Binary data.
    n_leaves : integer
        Number of leaves for each tree.
    n_trees : integer
        Number of trees to learn.
    step_size : float
        Shrinkage step size.
    use_gb : boolean
        Whether GradientBoosting should be used rather than LogitBoost.
    verbose : boolean
        Whether the currently trained dimension should be printed.

    Returns
    -------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    LLs: (n_dims,n_trees) float array
        Likelihoods for each dimension and number of trees.
    """

    assert X.dtype == np.bool
    X = np.float64(X)
    n_dims = X.shape[1]

    # train boosted trees for each dimension
    LLs = np.empty((n_dims, n_trees))
    model = n_dims * [None]
    for d in range(n_dims):
        if verbose: print(d, end=',', flush=True)
        model[d], LLs[d,] = logitboost.train(X[:,:d], X[:,d], n_leaves, n_trees, step_size=step_size, use_gb=use_gb)
    if verbose: print()

    return model, LLs


def log_likelihood(model, X, verbose=False):
    """
    Computes the likelihood of the data under the logitboost autoregressive network.

    Parameters
    ----------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    X : (n_samples,n_dims) array
        Binary data.
    verbose : boolean
        Whether the dimension currently being evaluated should be printed.

    Returns
    -------
    LLs : (n_samples,) float array
        Log-likelihoods.
    """

    n_samples, n_dims = X.shape
    assert len(model) == n_dims

    LLs = np.zeros(n_samples)
    for d in range(n_dims):
        if verbose: print(d, end=',', flush=True)
        LLs += logitboost.log_likelihood(model[d], X[:,:d], X[:,d])
    if verbose: print()

    return LLs


def log_likelihood_submodels(model, X):
    """
    Computes the likelihood for each dimension under each submodel (i.e. with fewer trees).

    Parameters
    ----------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    X : (n_samples,n_dims) array
        Binary data.

    Returns
    -------
    LLs : (n_dims,n_trees) float array
        Log-likelihoods.
    """

    n_samples, n_dims = X.shape
    assert len(model) == n_dims

    n_trees = len(model[-1])
    LLs = np.empty((n_dims,n_trees))
    for d in range(n_dims):
        LLs[d,] = logitboost.log_likelihood_submodels(model[d], X[:,:d], X[:,d])

    return LLs


def sample(model, n_samples, verbose=False):
    """
    Samples from the model.

    Parameters
    ----------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    n_samples : integer
        Number of samples.
    verbose : boolean
        Whether the currently sampled dimension should be printed.

    Returns
    -------
    X : (n_samples,n_dims) boolean array
        Sampled data.
    """

    n_dims = len(model)
    X = np.empty((n_samples, n_dims), dtype=np.bool)

    for d in range(n_dims):
        if verbose: print(d, end=',', flush=True)
        logitboost.sample(model[d], X[:,:d], X[:,d])
    if verbose: print()

    return X


def refit(model, X, step_size=1):
    """
    Updates the model parameters while keeping the tree structures fixed.

    Parameters
    ----------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    X : (n_samples,n_dims) array
        Binary data.
    step_size : float
        Shrinkage step size.

    Returns
    -------
    updated_model : (n_dims,) list of boosted trees
        Updated boosted trees for each dimension.
    """

    n_samples, n_dims = X.shape
    assert len(model) == n_dims

    # update parameters
    updated_model = n_dims * [None]
    for d in range(n_dims):
        updated_model[d] = logitboost.refit(model[d], X[:,:d], X[:,d], step_size=step_size)

    return updated_model


def individual_selection(model, LLs):
    """
    Selects the best number of leaves separately for each dimension.

    Parameters
    ----------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    LLs : (n_dims,n_trees) float array
        Log-likelihoods for each dimension and submodel.

    Returns
    -------
    selected_model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    """

    best_n_trees = 1 + LLs.argmax(1)
    return [model[d][:best_n_trees[d]] for d in range(len(model))]


def common_selection(model, LLs):
    """
    Selects the single best number of leaves for all dimensions.

    Parameters
    ----------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    LLs : (n_dims,n_trees) float array
        Log-likelihoods for each dimension and submodel.

    Returns
    -------
    selected_model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    """

    best_n_trees = 1 + LLs.sum(0).argmax()
    return [model[d][:best_n_trees] for d in range(len(model))]


def forward_selection(model, LLs_train, LLs_valid):
    """
    Sorts submodels for all dimensions through forward selection based on training performance.
    Selects the best model of that sequence based on validation performance.

    Parameters
    ----------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    LLs_train : (n_dims,n_trees) float array
        Training log-likelihoods for each dimension and submodel.
    LLs_valid : (n_dims,n_trees) float array
        Validation log-likelihoods for each dimension and submodel

    Returns
    -------
    selected_trees : (n_dims,) list of trees
        Trees for each dimension.
    """

    n_dims = len(model)
    n_trees = LLs_train.shape[1]

    # initialize
    LL_valid = LLs_valid[:,0].sum()
    best_LL_valid = LL_valid
    current_n_trees = np.ones(n_dims, dtype=np.int)
    best_n_trees = np.copy(current_n_trees)
    candidates = LLs_train[:,0] - LLs_train[:,1]
    candidates = [(candidates[d],d) for d in range(n_dims)]
    candidates.sort()

    # greedy selection
    while len(candidates) >= 1:
        d = candidates[0][1]
        LL_valid -= LLs_valid[d,current_n_trees[d]-1]
        current_n_trees[d] += 1
        LL_valid += LLs_valid[d,current_n_trees[d]-1]
        # replace candidate
        if current_n_trees[d] == n_trees:  # reached end
            heapq.heappop(candidates)
        else:
            heapq.heapreplace(candidates, (LLs_train[d,current_n_trees[d]-1]-LLs_train[d,current_n_trees[d]],d))
        if LL_valid > best_LL_valid:
            best_LL_valid = LL_valid
            best_n_trees[:] = current_n_trees

    return [model[d][:best_n_trees[d]] for d in range(len(model))]


def backward_selection(model, LLs_train, LLs_valid):
    """
    Sorts submodels for all dimensions through backward selection based on training performance.
    Selects the best model of that sequence based on validation performance.

    Parameters
    ----------
    model : (n_dims,) list of boosted trees
        Boosted trees for each dimension.
    LLs_train : (n_dims,n_trees) float array
        Training log-likelihoods for each dimension and submodel.
    LLs_valid : (n_dims,n_trees) float array
        Validation log-likelihoods for each dimension and submodel

    Returns
    -------
    selected_trees : (n_dims,) list of trees
        Trees for each dimension.
    """

    n_dims = len(model)
    n_trees = LLs_train.shape[1]

    # initialize
    LL_valid = LLs_valid[:,-1].sum()
    best_LL_valid = LL_valid
    current_n_trees = n_trees * np.ones(n_dims, dtype=np.int)
    best_n_trees = np.copy(current_n_trees)
    candidates = LLs_train[:,-1] - LLs_train[:,-2]
    candidates = [(candidates[d],d) for d in range(n_dims)]
    candidates.sort()

    # greedy selection
    while len(candidates) >= 1:
        d = candidates[0][1]
        LL_valid -= LLs_valid[d,current_n_trees[d]-1]
        current_n_trees[d] -= 1
        LL_valid += LLs_valid[d,current_n_trees[d]-1]
        # replace candidate
        if current_n_trees[d] == 1:  # reached end
            heapq.heappop(candidates)
        else:
            heapq.heapreplace(candidates, (LLs_train[d,current_n_trees[d]-1]-LLs_train[d,current_n_trees[d]-2],d))
        if LL_valid > best_LL_valid:
            best_LL_valid = LL_valid
            best_n_trees[:] = current_n_trees

    return [model[d][:best_n_trees[d]] for d in range(len(model))]
