from typing import Optional
import numpy as np
from itertools import product

def shuffle(arr:np.ndarray, seed:Optional[int]=None):
    shape = arr.shape

    ret = arr.copy().ravel()

    rng = np.random.default_rng(seed)
    idxs_perm = rng.permutation(arr.size)

    ret = ret[idxs_perm]

    return ret.reshape(shape), idxs_perm

def recover(arr:np.ndarray, idxs_perm):
    shape = arr.shape

    ret = arr.copy().ravel()

    ret = ret[np.argsort(idxs_perm)]

    return ret.reshape(shape)
    