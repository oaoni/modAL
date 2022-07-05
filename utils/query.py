"""
Functions to select certain element indices from an estimators current predictions.
"""
from scipy.sparse import coo_matrix, triu
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

def policy_func(policy):
    if policy == 'random':
        return random_query
    elif policy == 'max_uncertainty':
        return max_uncertainty
    elif policy == 'weighted_uncertainty':
        return weighted_uncertainty
    elif policy == 'max_guided_density':
        return max_guided_density
    elif policy == 'weighted_guided_density':
        return weighted_guided_density
    elif policy == 'max_guided_diversity':
        return max_guided_diversity
    elif policy == 'guided_exploration':
        return guided_exploration
    elif policy == 'global_recommendation':
        return global_recommendation
    elif policy == 'max_unique_uncertainty':
        return max_unique_uncertainty

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

def max_uncertainty(estimator, query_batch, is_sym, guide_data, guide_val):
    _, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_shape = X_test.shape

    query_rows, query_cols = active_sample(std,pred_row,pred_col,
                                           X_shape,'max',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def weighted_uncertainty(estimator, query_batch, is_sym, guide_data, guide_val):
    _, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_shape = X_test.shape

    query_rows, query_cols = active_sample(std,pred_row,pred_col,
                                           X_shape,'rand_prob',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

     

def random_query(estimator, query_batch, is_sym, guide_data, guide_val):
    _, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_shape = X_test.shape

    query_rows, query_cols = active_sample(X_test.data,pred_row,pred_col,
                                           X_shape,'rand',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def leverage_online(estimator, query_batch, is_sym, guide_data, guide_val):
    pred, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
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
                                           X_shape,'rand_prob',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def leverage_observed(estimator, query_batch, is_sym, guide_data, guide_val):
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
                                           X_shape,'rand_prob',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def max_guided_density(estimator, query_batch, is_sym, guide_data, guide_val):

    # Load guide df
    guide_df = estimator.guide_df

    X_test = estimator.X_testing.tocsc()
    test_coords = list(zip(*estimator.X_testing.tocsc().nonzero()))
    X_shape = X_test.shape

    # print('GUIDE DF HEAD: ', guide_df.head())
    # print('TEST COORDS: ', test_coords[:4])
    # print('FILT BOOL: ', guide_df['coords'].apply(eval).isin(test_coords)[:4])

    # Filter exploration guide df to only contain the present testing data[
    filt_guide = guide_df[guide_df['coords'].isin(test_coords)]

    # print('Filtered guide is: ', filt_guide.head(), 'with shape: ', filt_guide.shape)

    guide_values = filt_guide['density'].values
    guide_row = filt_guide['row_coord'].values
    guide_col = filt_guide['col_coord'].values

    query_rows, query_cols = active_sample(guide_values,guide_row,guide_col,
                                           X_shape,'max',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def weighted_guided_density(estimator, query_batch, is_sym, guide_data, guide_val):

    # Load guide df
    guide_df = estimator.guide_df

    X_test = estimator.X_testing.tocsc()
    test_coords = list(zip(*estimator.X_testing.tocsc().nonzero()))
    X_shape = X_test.shape

    # Filter exploration guide df to only contain the present testing data[
    filt_guide = guide_df[guide_df['coords'].isin(test_coords)]

    guide_values = filt_guide['density'].values
    guide_row = filt_guide['row_coord'].values
    guide_col = filt_guide['col_coord'].values

    query_rows, query_cols = active_sample(guide_values,guide_row,guide_col,
                                           X_shape,'rand_prob',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def max_guided_diversity(estimator, query_batch, is_sym, guide_data, guide_val):

    # Load guide df
    guide_df = estimator.guide_df

    X_test = estimator.X_testing.tocsc()
    test_coords = list(zip(*estimator.X_testing.tocsc().nonzero()))
    X_shape = X_test.shape
    guide_values = []
    guide_row = []
    guide_col = []

    sorted_guide_df = guide_df.copy().sort_values('density',ascending=False)
    sorted_guide_index = sorted_guide_df.index.values

    guide_size = guide_df.shape[0]
    w = 0 # Max Diversity sampling when w = 0
    quant = w/guide_size

    for i in range(query_batch):

        quant_similarity = sorted_guide_df['similarity'].quantile(quant)
        filt_guide = sorted_guide_df.query('similarity <= @quant_similarity')

        # Sample value from the candidate set
        filt_row = filt_guide.iloc[0,:][['row_coord', 'col_coord', 'coords','diversity']]
        sample_row,sample_col,coords,sample_val = filt_row
        filt_index = filt_row.name
        guide_row.append(sample_row)
        guide_col.append(sample_col)
        guide_values.append(sample_val)

        max_sim = np.maximum(guide_df['similarity'].values,
                             guide_data.iloc[:,filt_index].values)

        guide_df = guide_df\
        .assign(similarity=max_sim)
        # sorted_guide_df = sorted_guide_df\
        # .assign(similarity=max_sim[sorted_guide_index])
        sorted_guide_df = sorted_guide_df\
        .assign(similarity=guide_df['similarity'][sorted_guide_index])

    # Update guide df
    estimator.guide_df = guide_df
    estimator.query_dict['similarity'] = guide_df['similarity'].to_list()

    query_rows, query_cols = active_sample(guide_values,guide_row,guide_col,
                                           X_shape,'max',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def guided_exploration(estimator, query_batch, is_sym, guide_data, guide_val):

    # Load guide df
    guide_df = estimator.guide_df

    X_test = estimator.X_testing.tocsc()
    test_coords = list(zip(*estimator.X_testing.tocsc().nonzero()))
    X_shape = X_test.shape
    guide_values = []
    guide_row = []
    guide_col = []

    sorted_guide_df = guide_df.copy().sort_values('density',ascending=False)
    sorted_guide_index = sorted_guide_df.index.values

    guide_size = guide_df.shape[0]
    w = guide_val
    quant = w/guide_size

    for i in range(query_batch):

        quant_similarity = sorted_guide_df['similarity'].quantile(quant)
        filt_guide = sorted_guide_df.query('similarity <= @quant_similarity')

        # Sample value from the candidate set
        # Only guide row and col used, guide value is discarded
        filt_row = filt_guide.iloc[0,:][['row_coord', 'col_coord', 'coords','diversity']]
        sample_row,sample_col,coords,sample_val = filt_row
        filt_index = filt_row.name
        guide_row.append(sample_row)
        guide_col.append(sample_col)
        guide_values.append(sample_val)

        max_sim = np.maximum(guide_df['similarity'].values,
                             guide_data.loc[:,filt_index].values)


        guide_df = guide_df\
        .assign(similarity=max_sim)
        # sorted_guide_df = sorted_guide_df\
        # .assign(similarity=max_sim[sorted_guide_index])
        sorted_guide_df = sorted_guide_df\
        .assign(similarity=guide_df['similarity'][sorted_guide_index])


    # Update guide df
    estimator.guide_df = guide_df
    estimator.query_dict['similarity'] = guide_df['similarity'].to_list()

    query_rows, query_cols = active_sample(guide_values,guide_row,guide_col,
                                           X_shape,'max',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def global_recommendation(estimator, query_batch, is_sym, guide_data, guide_val):
    pred, coords = estimator.predict(return_std=False)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_shape = X_test.shape

    query_rows, query_cols = active_sample(np.abs(pred),pred_row,pred_col,
                                           X_shape,'max',query_batch, is_sym)

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data

def max_unique_uncertainty(estimator, query_batch, is_sym, guide_data, guide_val):

    _, std, coords = estimator.predict(return_std=True)
    pred_row,pred_col = zip(*coords)

    X_test = estimator.X_testing.tocsc()
    X_shape = X_test.shape

    data_sparse = coo_matrix((std,(pred_row,pred_col)), shape=X_shape)
    sort_inds = np.argsort(data_sparse.data)[::-1]

    sort_rows = np.array(pred_row)[sort_inds]
    sort_cols = np.array(pred_col)[sort_inds]

    unique_id = []
    query_rows = []
    query_cols = []

    for row,col in zip(sort_rows,sort_cols):

        if (row in unique_id) or (col in unique_id):
            pass
        else:
            query_rows += [row]
            query_cols += [col]
            unique_id += [row,col]

            if len(query_rows) == query_batch:
                break

    query_data = [X_test[row,col] for row,col in zip(query_rows, query_cols)]

    return query_rows, query_cols, query_data
