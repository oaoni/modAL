"""
Functions to select certain element indices from an estimators current predictions.
"""
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import random
from scipy.stats import entropy
from scipy.linalg import svd

# effective rank of matrix = entropy of singular value distribution
# one heuristic for choosing number of components to keep
def effective_rank(m):
    u, s, vt = svd(m)
    p = s/s.sum()
    return np.exp(entropy(p))

from sklearn.metrics.pairwise import pairwise_kernels

# row leverage scores of a matrix
# truncate svd to rank k, take left singular vectors, square, normalize by sum
# kernel is present in case you want to do kernelized version = svd on pairwise distance matrix
def rank_k_leverage_scores(m, k, kernel=None, **kwargs):
    if kernel is None:
        u, s, vt = svd(m)
    else:
        K = pairwise_kernels(m, metric=kernel, **kwargs)
        u, s, vt = svd(K)
    lev_scores = (u[:, :k]**2).sum(axis=1)
    if isinstance(m, pd.DataFrame):
        lev_scores = pd.Series(lev_scores, index=m.index)
    return lev_scores/lev_scores.sum()

# def produce a sampling distribution for a *symmetric* matrix
def sampling_distribution(m, k, kernel=None, **kwargs):
    lev_scores = rank_k_leverage_scores(m, k, kernel, **kwargs)
    lev_sampling = pd.DataFrame(np.outer(lev_scores, lev_scores),
                                index=lev_scores.index, columns=lev_scores.index)
    return lev_scores, lev_sampling

# predict which entry to read next based on leverage scores
# assumes matrix is symmetric, computes row leverage scores to produce a probability distrbituion on rows,
# and then produces a sampling distribution over the matrix by taking the outer product
# parameters: k = rank to reduce to when computing leverage scores
# (can potentially use effective_rank above or say effective_rank/2 as heuristics)
# n = number of indices to return
# returns flat indices that are always in the upper triangle of the matrix
def leverage_update(m, k, n=1, kernel=None, **kwargs):
    lev_scores, lev_sampling = sampling_distribution(m, k, kernel=None, **kwargs)
    # take only entries in upper triangle
    upper_tri_ind = np.triu_indices(lev_sampling.shape[0], k=1)
    #
    p = lev_sampling.values[upper_tri_ind]
    p = p/p.sum()
    flat_ind = np.ravel_multi_index(upper_tri_ind, lev_sampling.shape)
    if n == 1:
        return np.random.choice(flat_ind,n,replace=False, p=p)[0]
    else:
        return np.random.choice(flat_ind,n,replace=False, p=p)

def active_sample(data,row_ind,col_ind,shape,policy,query_batch,is_sym=False):
    """
    Samples a set of datapoints with the respective active learning policy
    Args:
    data: listed entries of the matrix
    row: row indices of the matrix entries
    col: col indices of the matrix entries
    shape: matrix shape
    policy: sampling policy to query matrix entries
    query_batch: int number of samples to query for batch
    is_sym: bool for sampling from a symmetrical matrix
    """

    data_sparse = coo_matrix((data,(row_ind,col_ind)),shape=shape)
    if is_sym:
        data_sparse = triu(data_sparse)

    if policy == 'max':
        query_idx = np.argpartition(data_sparse.data, -query_batch)[-query_batch:]
    elif policy == 'rand_prob':
        p = data_sparse.data/data_sparse.data.sum()
        query_idx = np.random.choice(len(p),size=query_batch,replace=False,p=p)
    elif policy == 'rand':
        num_tests = len(data_sparse.data)
        query_idx = random.sample(range(num_tests), query_batch)

    query_rows = data_sparse.row[query_idx]
    query_cols = data_sparse.col[query_idx]

    return query_rows, query_cols

def max_uncertainty(estimator, query_batch):
    _, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_shape = X_test.shape

    query_rows, query_cols = active_sample(std,pred_row,pred_col,
                                           X_shape,'max',query_batch)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def weighted_uncertainty(estimator, query_batch):
    _, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_shape = X_test.shape

    query_rows, query_cols = active_sample(std,pred_row,pred_col,
                                           X_shape,'rand_prob',query_batch)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def random_query(estimator, query_batch):
    _, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_shape = X_test.shape

    query_rows, query_cols = active_sample(X_test.data,pred_row,pred_col,
                                           X_shape,'rand',query_batch)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def leverage_online(estimator, query_batch):
    pred, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_shape = estimator.X_testing.shape
    X_pred = coo_matrix((pred, (pred_row, pred_col)), shape=X_shape).tocsc()

    #Computing leverage
    X_train = estimator.X_training.tocsc()
    X_all = X_pred + X_train

    target = pd.DataFrame(X_all.toarray())
    lev_scores, lev_sampling = sampling_distribution(target, int(effective_rank(target)/2))

    #Sample
    query_rows, query_cols = active_sample(lev_sampling.values[pred_row, pred_col],
                                           pred_row,pred_col,
                                           X_shape,'rand_prob',query_batch)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def leverage_observed(estimator, query_batch):
    _, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_train = estimator.X_training.tocsc()
    X_shape = X_test.shape
    X_all = X_test + X_train

    target = pd.DataFrame(X_all.toarray())
    lev_scores, lev_sampling = sampling_distribution(target, int(effective_rank(target)/2))

    test_lev = lev_sampling.values[pred_row, pred_col]

    #Active sample
    query_rows, query_cols = active_sample(test_lev,
                                           pred_row,pred_col,
                                           X_shape,'rand_prob',query_batch)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data
