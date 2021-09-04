from typing import Union, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


modALinput = Union[sp.csr_matrix, pd.DataFrame, np.ndarray, list]


def data_vstack(blocks: Sequence[modALinput]) -> modALinput:
    """
    Stack vertically sparse/dense arrays and pandas data frames.

    Args:
        blocks: Sequence of modALinput objects.

    Returns:
        New sequence of vertically stacked elements.
    """
    if any([sp.issparse(b) for b in blocks]):
        return sp.vstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        return blocks[0].append(blocks[1:])
    elif isinstance(blocks[0], np.ndarray):
        return np.concatenate(blocks)
    elif isinstance(blocks[0], list):
        return np.concatenate(blocks).tolist()

    raise TypeError('%s datatype is not supported' % type(blocks[0]))


def data_hstack(blocks: Sequence[modALinput]) -> modALinput:
    """
    Stack horizontally sparse/dense arrays and pandas data frames.

    Args:
        blocks: Sequence of modALinput objects.

    Returns:
        New sequence of horizontally stacked elements.
    """
    if any([sp.issparse(b) for b in blocks]):
        return sp.hstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        pd.concat(blocks, axis=1)
    elif isinstance(blocks[0], np.ndarray):
        return np.hstack(blocks)
    elif isinstance(blocks[0], list):
        return np.hstack(blocks).tolist()

    TypeError('%s datatype is not supported' % type(blocks[0]))


def add_row(X:modALinput, row: modALinput):
    """
    Returns X' =

    [X

    row]
    """
    if isinstance(X, np.ndarray):
        return np.vstack((X, row))
    elif isinstance(X, list):
        return np.vstack((X, row)).tolist()

    # data_vstack readily supports stacking of matrix as first argument
    # and row as second for the other data types
    return data_vstack([X, row])


def retrieve_rows(X: modALinput,
                  I: Union[int, List[int], np.ndarray]) -> Union[sp.csc_matrix, np.ndarray, pd.DataFrame]:
    """
    Returns the rows I from the data set X

    For a single index, the result is as follows:
    * 1xM matrix in case of scipy sparse NxM matrix X
    * pandas series in case of a pandas data frame
    * row in case of list or numpy format
    """
    if sp.issparse(X):
        # Out of the sparse matrix formats (sp.csc_matrix, sp.csr_matrix, sp.bsr_matrix,
        # sp.lil_matrix, sp.dok_matrix, sp.coo_matrix, sp.dia_matrix), only sp.bsr_matrix, sp.coo_matrix
        # and sp.dia_matrix don't support indexing and need to be converted to a sparse format
        # that does support indexing. It seems conversion to CSR is currently most efficient.

        try:
            return X[I]
        except:
            sp_format = X.getformat()
            return X.tocsr()[I].asformat(sp_format)
    elif isinstance(X, pd.DataFrame):
        return X.iloc[I]
    elif isinstance(X, np.ndarray):
        return X[I]
    elif isinstance(X, list):
        return np.array(X)[I].tolist()

    raise TypeError('%s datatype is not supported' % type(X))


def drop_rows(X: modALinput,
              I: Union[int, List[int], np.ndarray]) -> Union[sp.csc_matrix, np.ndarray, pd.DataFrame]:
    """
    Returns X without the row(s) at index/indices I
    """
    if sp.issparse(X):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[I] = False
        return retrieve_rows(X, mask)
    elif isinstance(X, pd.DataFrame):
        return X.drop(I, axis=0)
    elif isinstance(X, np.ndarray):
        return np.delete(X, I, axis=0)
    elif isinstance(X, list):
        return np.delete(X, I, axis=0).tolist()

    raise TypeError('%s datatype is not supported' % type(X))


def enumerate_data(X: modALinput):
    """
    for i, x in enumerate_data(X):

    Depending on the data type of X, returns:

    * A 1xM matrix in case of scipy sparse NxM matrix X
    * pandas series in case of a pandas data frame X
    * row in case of list or numpy format
    """
    if sp.issparse(X):
        return enumerate(X.tocsr())
    elif isinstance(X, pd.DataFrame):
        return X.iterrows()
    elif isinstance(X, np.ndarray) or isinstance(X, list):
        # numpy arrays and lists can readily be enumerated
        return enumerate(X)

    raise TypeError('%s datatype is not supported' % type(X))


def data_shape(X: modALinput):
    """
    Returns the shape of the data set X
    """
    if sp.issparse(X) or isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
        # scipy.sparse, pandas and numpy all support .shape
        return X.shape
    elif isinstance(X, list):
        return np.array(X).shape

    raise TypeError('%s datatype is not supported' % type(X))

def addData2matrix(X_train: modALinput, X_new, X_idx, symmetrical) -> modALinput:
    """
    Adds the new data to a sparse matrix

    Args:
        X_train: Training data in sparse format matrix
        X_new: Value of new data to be added to training set
        X_idx: Set of form (sparse_coordinate, row_idx, col_idx)

    Returns:
        New sequence of vertically stacked elements.
    """

    data = X_new
    row = X_idx[1]
    col = X_idx[2]

    if (symmetrical) and (row != col):
        X_new = sp.coo_matrix(([data,data], ([row, col], [col, row])),shape=X_train.shape)
    else:
        X_new = sp.coo_matrix(([data], ([row], [col])),shape=X_train.shape)

    return sp.coo_matrix(X_train + X_new)

def removeData2matrix(X_test: modALinput, X_idx, symmetrical) -> modALinput:
    """
    Removes the queried data from the sparse testing matrix

    Args:
        X_test: Testing data in sparse format matrix
        X_idx: Set of form (sparse_coordinate, row_idx, col_idx)

    Returns:
        New sequence of vertically stacked elements.
    """
    if (symmetrical) and (X_idx[1] != X_idx[2]):
        row = X_idx[1]
        col = X_idx[2]
        mask = sp.coo_matrix(([-1,-1], ([row,col],[col,row])),shape=X_test.shape) + sp.coo_matrix(np.ones(X_test.shape))

        X_new = sp.coo_matrix(X_test.multiply(mask))

    else:
        data = np.delete(X_test.data,X_idx[0])
        row = np.delete(X_test.row,X_idx[0])
        col = np.delete(X_test.col,X_idx[0])

        X_new = sp.coo_matrix((data, (row, col)),shape=X_test.shape)



    return X_new
