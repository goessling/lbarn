
#
# LogitBoost for binary features.
#

import numpy as np

import regression_tree



def train(X, y, n_leaves, n_trees, step_size=1, use_gb=False):
    """
    Trains LogitBoost.

    Parameters
    ----------
    X : (n_samples,n_features) array
        Binary features.
    y : (n_samples,) array
        Binary labels.
    n_leaves : integer
        Number of leaves for each tree.
    n_trees : integer
        Number of trees to learn.
    step_size : float
        Shrinkage step size.
    use_gb : boolean
        Whether GradientBoosting should be used rather than LogitBoost.

    Returns
    -------
    trees : (n_trees,) list of regression trees
        (parent,prediction,feature)
    LLs : (n_trees,) float array
        Likelihoods of models.
    """

    X = np.float64(X)
    y = 2. * y - 1  # convert to -/+ 1

    # initialize
    log_odds = np.zeros(X.shape[0])

    # fit boosted trees
    trees = n_trees * [None]
    LLs = np.empty(n_trees)
    for t in range(n_trees):
        residuals = y / (1+np.exp(y*log_odds))  # y - p
        abs_residuals = np.abs(residuals)
        weights = abs_residuals*(1-abs_residuals)  # p(1-p)
        # train tree on residuals
        if use_gb:
            tree = regression_tree.train(X, residuals, n_leaves)[0]  # unweighted regression
        else:
            tree = regression_tree.train_robust(X, residuals, weights, n_leaves)[0]  # weighted regression
        # update predictions
        for node in range(len(tree)):
            if tree[node][2] == -1:  # is leaf
                subset = tree[node][3]
                normalization = weights[subset].sum()
                if normalization == 0:
                    value = 0
                else:
                    value = residuals[subset].sum() / normalization
                value *= step_size
                tree[node] = (tree[node][0], value, -1)
                log_odds[subset] += value
        trees[t] = tree
        LLs[t] = -np.log(1+np.exp(-y*log_odds)).sum()  # log(p) if y pos and log(1-p) if y neg

    return trees, LLs


def predict(trees, X):
    """
    Predicts the log-odds for the feature vectors.

    Parameters
    ----------
    trees : List of regression trees
        (parent,prediction,feature)
    X : (n_samples,n_features) array
        Binary features.

    Returns
    -------
    predictions : (n_samples,) float array
        Predictions for the feature vectors.
    """

    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    n_trees = len(trees)
    for t in range(n_trees):
        predictions += regression_tree.predict(trees[t], X)

    return predictions


def log_likelihood(trees, X, y):
    """
    Computes the likelihoods under the (full) model.

    Parameters
    ----------
    trees : List of regression trees
        (parent,prediction,feature)
    X : (n_samples,n_features) array
        Binary features.
    y : (n_samples,) array
        Binary labels.

    Returns
    -------
    LLs : (nsample,) float array
        Log-likelihoods.
    """

    y = 2. * y - 1  # convert to -/+ 1
    return -np.log(1+np.exp(-y*predict(trees, X)))


def log_likelihood_submodels(trees, X, y):
    """
    Computes the log-likelihood under each submodel (i.e. with fewer trees).

    Parameters
    ----------
    trees : (n_trees,) list of regression trees
        (parent,prediction,feature)
    X : (n_samples,n_features) boolean array
        Binary features.
    y : (n_samples,) boolean array
        Binary labels.

    Returns
    -------
    LLs : (n_trees,) float array
        Likelihoods of models.
    """

    y = 2. * y - 1  # convert to -/+ 1

    n_trees = len(trees)
    LLs = np.empty(n_trees)
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)

    for t in range(n_trees):
        predictions += regression_tree.predict(trees[t], X)
        LLs[t] = -np.log(1+np.exp(-y*predictions)).sum()

    return LLs


def probabilities(trees, X):
    """
    Predicts the probabilities for the feature vectors.

    Parameters
    ----------
    trees : (n_trees,) list of regression trees
        (parent,prediction,feature)
    X : (n_samples,n_features) boolean array
        Binary features.

    Returns
    -------
    probabilities : (n_samples,) float array
        Success probabilities for the feature vectors.
    """

    return 1 / (1+np.exp(-predict(trees,X)))


def sample(trees, X, y):
    """
    Samples the from the LogitBoost model.

    Parameters
    ----------
    trees : (n_trees,) list of regression trees
        (parent,prediction,feature)
    X : (n_samples,n_features) boolean array
        Binary features.
    y : (n_samples,) boolean array
        Output buffer.
    """

    nsample = X.shape[0]
    y[:] = np.random.rand(nsample) <= probabilities(trees, X)


def refit(trees, X, y, step_size=1):
    """
    Updates the parameters while keeping the tree structures fixed.

    Parameters
    ----------
    trees : (n_trees,) list of regression trees
        (parent,prediction,feature)
    X : (n_samples,n_features) boolean array
        Binary features.
    y : (n_samples,) [float] array
        Binary labels.
    step_size : float
        Shrinkage step size.

    Returns
    -------
    updated_trees : (n_trees,) list of regression trees
        Updated regression trees.
    """

    X = np.float64(X)
    y = 2. * y - 1  # convert to -/+ 1

    # initialize
    log_odds = np.zeros(X.shape[0])

    # refit trees
    n_trees = len(trees)
    updated_trees = n_trees * [None]
    for t in range(n_trees):
        tree = trees[t]
        subsets = regression_tree.partition(tree, X)
        n_nodes = len(tree)
        updated_tree = n_nodes * [None]
        residuals = y / (1+np.exp(y*log_odds))  # y - p
        abs_residuals = np.abs(residuals)
        weights = abs_residuals*(1-abs_residuals)  # p(1-p)
        # update predictions
        for node in range(n_nodes):
            if tree[node][2] == -1:  # is leaf
                subset = subsets[node]
                normalization = weights[subset].sum()
                if normalization == 0:
                    value = 0
                else:
                    value = residuals[subset].sum() / normalization
                value *= step_size
                updated_tree[node] = (tree[node][0], value, -1)
                log_odds[subset] += value
            else:
                updated_tree[node] = tree[node]
        updated_trees[t] = updated_tree

    return updated_trees
