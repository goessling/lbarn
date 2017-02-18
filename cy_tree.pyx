#cython: boundscheck=False
#cython: wraparound=False

#
# Fast implementation of splitting operation.
#


import numpy as np
cimport numpy as cnp



def subset_split(cnp.ndarray[Py_ssize_t,mode='c',ndim=1] ind_arr, cnp.ndarray[cnp.uint8_t,ndim=1,mode='c',cast=True] x):
    """
    Splits the index set according to the binary feature.

    Parameters
    ----------
    ind_arr : (n_samples,) integer array
        Indices
    x : (n_samples,) boolean array
        Binary feature values.

    Returns
    -------
    subset_on : (n_samples_on,) integer array
        Indices with positive feature value.
    subset_on : (n_samples_off,) integer array
        Indices with negative feature value.
    """
    
    # create buffers
    cdef Py_ssize_t n_samples = ind_arr.size
    cdef cnp.ndarray[Py_ssize_t,ndim=1,mode='c'] subset_on = np.empty(n_samples, dtype=np.int)
    cdef cnp.ndarray[Py_ssize_t,ndim=1,mode='c'] subset_off = np.empty(n_samples, dtype=np.int)
    
    # split indices
    cdef Py_ssize_t non = 0
    cdef Py_ssize_t noff = 0
    cdef Py_ssize_t s
    for s in range(n_samples):
        if x[s]:
            subset_on[non] = ind_arr[s]
            non += 1
        else:
            subset_off[noff] = ind_arr[s]
            noff += 1
    
    return subset_on[:non], subset_off[:noff]
