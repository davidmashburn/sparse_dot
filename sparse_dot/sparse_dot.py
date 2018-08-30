'''The main script'''
from __future__ import print_function
from __future__ import absolute_import
from builtins import map

# To update with any Cython changes, just run:
# python setup.py build_ext --inplace

import numpy as np
from . import cy_sparse_dot

try:
    import scipy.sparse
except ImportError:
    print('Scipy not found, scipy.sparse functionality will not be available')

def to_saf(arr1d):
    arr1d = np.asanyarray(arr1d)
    locs = np.nonzero(arr1d)
    return {'locs': locs[0].astype(np.uint32),
            'array': arr1d[locs].astype(np.float32)}

def to_saf_list(arr2d):
    return list(map(to_saf, arr2d))

def saf_list_to_csr_matrix(saf_list, shape):
    data = np.concatenate([saf['array'] for saf in saf_list])
    indices = np.concatenate([saf['locs'] for saf in saf_list])
    indptr = np.cumsum([0] + [len(saf['array']) for saf in saf_list])
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

def validate_saf(saf, verbose=True):
    '''True if the locs (indices) in a saf are ordered
       AND the data types of the arrays are uint32 and float32 respectively'''
    def vpr(x):
        if verbose:
            print(x)
    if not ('locs' in saf and 'array' in saf):
        vpr('missing members')
        return False
    if not (hasattr(saf['locs'], 'dtype') and
            hasattr(saf['array'], 'dtype')):
        vpr('members not arrays')
        return False
    if not (saf['locs'].dtype == np.uint32 and
            saf['array'].dtype == np.float32):
        vpr('bad dtype')
        return False
    if not np.all(saf['locs'][1:] > saf['locs'][:-1]):
        vpr('locs not ordered')
        return False
    
    return True

def sparse_dot_full(saf_list, validate=True, verbose=True):
    '''Takes a list of arrays in locs/array dict form and '''
    if validate:
        assert all(validate_saf(saf, verbose=verbose) for saf in saf_list)
    return cy_sparse_dot.cy_sparse_dot_full(saf_list)

def dot_full_using_sparse(arr):
    '''Takes a 2d array and runs dot products against every
       combination of rows'''
    return sparse_dot_full(to_saf_list(arr), validate=False)

def sparse_cos_similarity(saf_list, validate=True, verbose=True):
    norms = np.array([np.linalg.norm(saf['array']) for saf in saf_list])
    dots = sparse_dot_full(saf_list, validate=validate, verbose=verbose)
    norm_i, norm_j = norms[(dots['i'],)], norms[(dots['j'],)]
    dots['sparse_result'] /= norm_i * norm_j
    return dots

def sparse_cos_distance(saf_list, validate=True, verbose=True):
    dots = sparse_cos_similarity(saf_list, validate=validate, verbose=verbose)
    dots['sparse_result'] *= -1
    dots['sparse_result'] += 1
    return dots

def cos_similarity_using_sparse(arr):
    return sparse_cos_similarity(to_saf_list(arr))

def cos_distance_using_sparse(arr):
    return sparse_cos_distance(to_saf_list(arr))

def coo_cosine_similarity(input_coo_matrix):
    sq = lambda x: x * x.T
    output_csr_matrix = input_coo_matrix.tocsr()
    sqrt_sum_square_rows = np.array(np.sqrt(sq(output_csr_matrix).sum(axis=1)))[:, 0]
    output_csr_matrix.data /= sqrt_sum_square_rows[input_coo_matrix.row]
    return sq(output_csr_matrix)

def csr_cosine_similarity(input_csr_matrix):
    similarity = input_csr_matrix * input_csr_matrix.T
    square_mag = similarity.diagonal()
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    return similarity.multiply(inv_mag).T.multiply(inv_mag)

if __name__ == '__main__':
    r = dot_full_using_sparse([[1, 0, 0, 1, 3, 1],
                               [2, 0, 0, 0, 1, 5]])
    print(r)
