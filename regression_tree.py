
#
# Regression tree for binary features.
#

import heapq

import numpy as np

try:
    from cy_tree import subset_split
except ImportError:
    def subset_split(ind_arr, x):
        subset_on = ind_arr[x]
        subset_off = np.setdiff1d(ind_arr, subset_on, assume_unique=True)
        return subset_on, subset_off
    print('slow implementation')



def train(X, y, n_leaves, weights=None):
    """
    Trains a regression tree in a greedy manner.

    Parameters
    ----------
    X : (n_samples,n_features) array
        Binary features.
    y : (n_samples,) float array
        Response values.
    n_leaves : integer
        Number of leaves to learn.
    weights : (n_samples,) float array
        Sample weights.

    Returns
    -------
    tree : (<=2*n_leaves-1,) list of nodes
        (parent,prediction,feature[,subset])
    RSS : (n_leaves,) float array
        Weighted sums of squares.
    """

    X = np.float64(X)
    n_samples, n_features = X.shape

    # create root
    if weights is None:
        root_sum = y.sum()
        root_normalization = y.size
    else:
        root_sum = weights.dot(y)
        root_normalization = weights.sum()
    prediction_root = root_sum / root_normalization
    RSS = np.empty(n_leaves)
    RSS[0] = -root_sum**2 / root_normalization
    tree = [(-1,prediction_root,-1,np.arange(n_samples))]
    # tree: (parent,prediction,feature[,subset]), parent is -1 for the root, feature is -1 for leaves, subset only for leaves

    # create root split candidate
    if n_features != 0:
        split_root = best_split(X, y, weights=weights)
        RSS_split = split_root[3] + split_root[6]
        candidates = [(RSS_split-RSS[0],0)+split_root]
        # candidates: (RSS-RSS_parent,parent,feature,prediction_on,subset_on,RSS_on,prediction_off,subset_off,RSS_off)

    # refine partition
    nsplit = min(n_leaves, 2**n_features) - 1
    for s in range(nsplit):
        # perform best split
        RSS_diff, parent, feature, prediction_on, subset_on, RSS_on, prediction_off, subset_off, RSS_off = candidates[0]
        RSS[s+1] = RSS[s] + RSS_diff

        # make parent into inner node
        tree[parent] = tree[parent][:2] + (feature,)

        # add 2 leaves
        tree.append((parent, prediction_on, -1, subset_on))
        tree.append((parent, prediction_off, -1, subset_off))

        # add split candidates for leaves
        if s != nsplit-1:  # not last split
            split_on = best_split(X, y, subset=subset_on, weights=weights)
            RSS_split = split_on[3] + split_on[6]
            heapq.heapreplace(candidates, (RSS_split-RSS_on,len(tree)-2)+split_on)
            split_off = best_split(X, y, subset=subset_off, weights=weights)
            RSS_split = split_off[3] + split_off[6]
            heapq.heappush(candidates, (RSS_split-RSS_off,len(tree)-1)+split_off)

    RSS[nsplit+1:] = RSS[nsplit]

    # add global constant
    if weights is None:
        RSS += y.dot(y)
    else:
        RSS += weights.dot(y**2)

    return tree, RSS


def best_split(X, y, subset=None, weights=None):
    """
    Finds the best split of the data subset.

    Parameters
    ----------
    X : (n_samples,n_features) float array
        Binary features.
    y : (n_samples,) float array
        Response values.
    subset : integer array
        Indices. All if None.
    weights : (n_samples,) float array
        Sample weights. Ones if None.

    Returns
    -------
    out : (integer,float,integer array,float,float,integer_array,float)
        feature, prediction_on, subset_on, sum_of_squares_on, prediction_off, subset_off, sum_of_squares_off
    """

    if subset is not None:  # takes about 50% of runtime, rest is dot and sum (0-padding not an option since subset may be small)
        X = X[subset,]
        y = y[subset]
        if weights is not None:
            weights = weights[subset]
    else:
        subset = np.arange(X.shape[0])
    n_features = X.shape[1]

    # compute counts
    if weights is None:
        n_features_on = X.sum(0)
    else:
        n_features_on = weights.dot(X)
        y = weights * y
    feature_on_sums = y.dot(X)
    if weights is None:
        n_features_off = y.size - n_features_on
    else:
        n_features_off = weights.sum() - n_features_on
    feature_off_sums = y.sum() - feature_on_sums

    # compute RSS, up to a global constant
    # (the RSS of a split is: sum y**2 - (sum_on y)**2/non - (sum_off y)**2/noff)
    ii = np.where(n_features_on>0)[0]
    RSS_on = np.zeros(n_features)
    RSS_on[ii] = -feature_on_sums[ii]**2 / n_features_on[ii]
    ii = np.where(n_features_off>0)[0]
    RSS_off = np.zeros(n_features)
    RSS_off[ii] = -feature_off_sums[ii]**2 / n_features_off[ii]
    RSS = RSS_on + RSS_off

    # find best feature
    feature = RSS.argmin()
    if n_features_on[feature]>0:
        prediction_on = feature_on_sums[feature] / n_features_on[feature]
    else:
        prediction_on = 0
    if n_features_off[feature]>0:
        prediction_off = feature_off_sums[feature] / n_features_off[feature]
    else:
        prediction_off = 0
    subset_on, subset_off = subset_split(subset, np.bool8(X[:,feature]))

    return feature, prediction_on, subset_on, RSS_on[feature], prediction_off, subset_off, RSS_off[feature]


def train_robust(X, y, weights, n_leaves):
    """
    Trains a regression tree for y/w on x with weights w in a robust way.

    Parameters
    ----------
    X : (n_samples,n_features) array
        Binary features.
    y : (n_samples,) float array
        Response values.
    weights : (n_samples,) float array
        Sample weights.
    n_leaves : integer
        Number of splits to perform.

    Returns
    -------
    tree : (<=2*n_leaves-1,) list of nodes
        (parent,prediction,feature[,subset])
    RSS : (n_leaves,) float array
        Weighted sums of squares, up to the global constant sum y**2/w.
    """

    X = np.float64(X)
    n_samples, n_features = X.shape

    # create root
    root_sum = y.sum()
    root_normalization = weights.sum()
    prediction_root = root_sum / root_normalization
    RSS = np.inf * np.ones(n_leaves)
    RSS[0] = -root_sum**2 / root_normalization
    tree = [(-1,prediction_root,-1,np.arange(n_samples))]
    # tree: (parent,prediction,feature[,subset]), parent is -1 for the root, feature is -1 for leaves, subset only for leaves

    # create root split candidate
    if n_features != 0:
        split_root = best_split_robust(X, y, weights)
        RSS_split = split_root[3] + split_root[6]
        candidates = [(RSS_split-RSS[0],0)+split_root]
        # candidates: (RSS-RSS_parent,parent,feature,prediction_on,subset_on,RSS_on,prediction_off,subset_off,RSS_off)

    # refine partition
    n_splits = min(n_leaves, 2**n_features) - 1
    for s in range(n_splits):
        # perform best split
        RSS_diff, parent, feature, prediction_on, subset_on, RSS_on, prediction_off, subset_off, RSS_off = candidates[0]
        RSS[s+1] = RSS[s] + RSS_diff

        # make parent into inner node
        tree[parent] = tree[parent][:2] + (feature,)

        # add 2 leaves
        tree.append((parent, prediction_on, -1, subset_on))
        tree.append((parent, prediction_off, -1, subset_off))

        # add split candidates for leaves
        if s != n_splits-1:  # not last split
            split_on = best_split_robust(X, y, weights, subset=subset_on)
            RSS_split = split_on[3] + split_on[6]
            heapq.heapreplace(candidates, (RSS_split-RSS_on,len(tree)-2)+split_on)
            split_off = best_split_robust(X, y, weights, subset=subset_off)
            RSS_split = split_off[3] + split_off[6]
            heapq.heappush(candidates, (RSS_split-RSS_off,len(tree)-1)+split_off)

    return tree, RSS


def best_split_robust(X, y, weights, subset=None):
    """
    Finds the best split of the data subset for the weighted regression problem.

    Parameters
    ----------
    X : (n_samples,n_features) float array
        Binary features.
    y : (n_samples,) float array
        Response values.
    weights : (n_samples,) float array
        Sample weights.
    subset : integer array
        Indices. All if None.

    Returns
    -------
    out : (integer,float,integer array,float,float,integer_array,float)
        feature, prediction_on, subset_on, sum_of_squares_on, prediction_off, subset_off, sum_of_squares_off
    """

    if subset is not None:  # takes about 50% of runtime, rest is dot and sum (0-padding not an option since subset may be small)
        X = X[subset,]
        y = y[subset]
        weights = weights[subset]
    else:
        subset = np.arange(X.shape[0])
    n_features = X.shape[1]

    # compute counts
    n_features_on = weights.dot(X)
    feature_on_sum = y.dot(X)
    n_features_off = weights.sum() - n_features_on
    feature_off_sum = y.sum() - feature_on_sum

    # compute RSS, up to a global constant
    ii = np.where(n_features_on>0)[0]
    RSS_on = np.zeros(n_features)
    RSS_on[ii] = -feature_on_sum[ii]**2 / n_features_on[ii]
    ii = np.where(n_features_off>0)[0]
    RSS_off = np.zeros(n_features)
    RSS_off[ii] = -feature_off_sum[ii]**2 / n_features_off[ii]
    RSS = RSS_on + RSS_off

    # find best feature
    feature = RSS.argmin()
    if n_features_on[feature]>0:
        prediction_on = feature_on_sum[feature] / n_features_on[feature]
    else:
        prediction_on = 0
    if n_features_off[feature]>0:
        prediction_off = feature_off_sum[feature] / n_features_off[feature]
    else:
        prediction_off = 0
    subset_on, subset_off = subset_split(subset, np.bool8(X[:,feature]))

    return feature, prediction_on, subset_on, RSS_on[feature], prediction_off, subset_off, RSS_off[feature]


def predict(tree, X):
    """
    Predicts the responses for the feature vectors.

    Parameters
    ----------
    tree : (n_nodes,) list of nodes
        (parent,prediction,feature)
    X : (n_samples,n_features) [boolean] array
        Binary features.

    Returns
    -------
    y : (n_samples,) float array
        Responses.
    """

    n_nodes = len(tree)
    n_splits = n_nodes // 2

    # find indices for each node
    subsets = n_nodes * [None]
    subsets[0] = np.arange(X.shape[0])
    for s in range(n_splits):
        parent = tree[1+2*s][0]
        feature = tree[parent][2]
        subset = subsets[parent]
        subsets[1+2*s], subsets[1+2*s+1] = subset_split(subset, np.bool8(X[subset,feature]))

    # store prediction for all samples
    y = np.empty(X.shape[0])
    for node in range(n_nodes):
        if tree[node][2] == -1:  # is leaf
            y[subsets[node]] = tree[node][1]
    return y


def partition(tree, X):
    """
    Partitions the data according to the tree splits.

    Parameters
    ----------
    tree : (n_nodes,) list of nodes
        (parent,prediction,feature)
    X : (n_samples,n_features) [boolean] array
        Binary features.

    Returns
    -------
    subsets : (n_nodes,) list of integer arrays
        Indices for each leaf, None for inner nodes.
    """

    n_nodes = len(tree)
    n_splits = n_nodes // 2

    # find indices for each node
    subsets = n_nodes * [None]
    subsets[0] = np.arange(X.shape[0])
    for s in range(n_splits):
        parent = tree[1+2*s][0]
        feature = tree[parent][2]
        subset = subsets[parent]
        subsets[1+2*s], subsets[1+2*s+1] = subset_split(subset, np.bool8(X[subset,feature]))
        subsets[parent] = None

    return subsets
